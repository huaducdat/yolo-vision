from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4
import logging
import time

import cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, HTMLResponse
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_FORMATS = {
    ".jpg": "JPEG",
    ".png": "PNG",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ghi log vào file detection.log.
logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.FileHandler(
        BASE_DIR / "detection.log",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model một lần khi khởi động API.
    app.state.model = YOLO(str(BASE_DIR / "yolov8n.pt"))
    app.state.segment_model = YOLO(str(BASE_DIR / "yolov8n-seg.pt"))
    app.state.model_lock = Lock()
    yield


app = FastAPI(
    title="YOLO Object Detection API",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
def home():
    return HTMLResponse(r"""
    <!DOCTYPE html>
    <html lang="vi">
        <head>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="1;url=/docs">
            <title>YOLO API</title>
        </head>
        <body>
            <p>YOLO API đang chạy. Chuyển đến Swagger sau 1 giây...</p>
            <a href="/docs">Mở ngay</a>
        </body>
    </html>
    """)

@app.get("/classes")
def get_classes(request: Request):
    """Trả danh sách các lớp mà model có thể nhận diện."""
    names = request.app.state.model.names

    return {
        "total": len(names),
        "classes": [
            {"id": class_id, "name": name}
            for class_id, name in sorted(names.items())
        ],
    }


@app.post(
    "/upload",
    response_class=Response,
    responses={
        200: {
            "description": "Ảnh JPEG đã vẽ bounding boxes",
            "content": {"image/jpeg": {}},
        }
    },
)
def upload_image(
    request: Request,
    file: UploadFile = File(...),
    crop_person: bool = False,
):
    started_at = time.perf_counter()
    temp_path = None

    try:
        # 1. Kiểm tra phần mở rộng.
        extension = Path(file.filename or "").suffix.lower()

        if extension not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail="Chỉ chấp nhận file .jpg hoặc .png.",
            )

        # 2. Kiểm tra dung lượng.
        content = file.file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File không được lớn hơn 5 MB.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="File ảnh rỗng.",
            )

        # 3. Kiểm tra nội dung thật của ảnh.
        # Đổi tên file văn bản thành .jpg vẫn không qua được bước này.
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format != ALLOWED_FORMATS[extension]:
                    raise HTTPException(
                        status_code=400,
                        detail="Nội dung ảnh không khớp với đuôi file.",
                    )

                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail="Ảnh không hợp lệ hoặc bị hỏng.",
            ) from exc

        # 4. Lưu ảnh tạm bằng tên ngẫu nhiên, tránh trùng tên.
        temp_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
        temp_path.write_bytes(content)

        # 5. Nhận diện và vẽ bounding boxes, hoặc cắt ảnh.
        # Lock tránh nhiều request dùng cùng model đồng thời.
        with request.app.state.model_lock:
            model = (request.app.state.segment_model if crop_person
                     else request.app.state.model)
            options = {"retina_masks": True} if crop_person else {}
            result = model.predict(
                source=str(temp_path), conf=0.25, device="cpu",
                verbose=False, **options,
            )[0]

            if crop_person:
                indices = [
                    i for i, box in enumerate(result.boxes)
                    if result.names[int(box.cls.item())] == "person"
                ]
                if not indices or result.masks is None:
                    raise HTTPException(404, "Không tìm thấy người để tách nền.")

                def area(index):
                    x1, y1, x2, y2 = result.boxes[index].xyxy[0].tolist()
                    return (x2 - x1) * (y2 - y1)

                selected = max(indices, key=area)
                mask = result.masks.data[selected].cpu().numpy()
                alpha = (mask > 0.5).astype("uint8") * 255
                original = result.orig_img
                if alpha.shape != original.shape[:2]:
                    raise RuntimeError("Kích thước mask không khớp ảnh.")
                points = cv2.findNonZero(alpha)
                if points is None:
                    raise HTTPException(404, "Mask của người rỗng.")
                transparent = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
                transparent[:, :, 3] = alpha
                x, y, w, h = cv2.boundingRect(points)
                annotated_image = transparent[y:y+h, x:x+w].copy()
            else:
                annotated_image = result.plot()

            object_count = len(result.boxes)
            detected_names = [
                result.names[int(class_id)]
                for class_id in result.boxes.cls.tolist()
            ]

        extension_out = ".png" if crop_person else ".jpg"
        parameters = [] if crop_person else [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, encoded_image = cv2.imencode(
            extension_out, annotated_image, parameters,
        )
        if not success:
            raise RuntimeError("Không thể mã hóa ảnh kết quả.")

        # 7. Ghi log lần xử lý thành công.
        elapsed = time.perf_counter() - started_at

        logger.info(
            "mode=%s | image=%s | bytes=%d | objects=%d | classes=%s | seconds=%.2f",
            "segment-person" if crop_person else "detect",
            temp_path.name,
            len(content),
            object_count,
            ",".join(detected_names) or "none",
            elapsed,
        )

        # 8. Trả trực tiếp dữ liệu ảnh.
        return Response(
            content=encoded_image.tobytes(),
            media_type="image/png" if crop_person else "image/jpeg",
            headers={
                "Content-Disposition": ('inline; filename="person.png"'
                    if crop_person else 'inline; filename="detected.jpg"')
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Lỗi xử lý ảnh")

        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý ảnh. Xem detection.log.",
        ) from exc

    finally:
        # Luôn đóng file và dọn ảnh tạm, kể cả khi xử lý lỗi.
        file.file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

@app.post(
    "/crop-person",
    response_class=Response,
    responses={
        200: {
            "description": "Ảnh người đã tách nền PNG",
            "content": {"image/png": {}},
        }
    },
)
def crop_person_api(
    request: Request,
    file: UploadFile = File(...),
):
    return upload_image(
        request=request,
        file=file,
        crop_person=True,
    )

@app.get("/crop-person", response_class=HTMLResponse)
def crop_person_page():
    return HTMLResponse(r"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Cắt người trong ảnh</title>

        <style>
            body {
                font-family: system-ui, sans-serif;
                background: #f4f6fa;
                color: #182238;
                margin: 0;
                padding: 32px 16px;
            }

            main {
                max-width: 960px;
                margin: auto;
                padding: 28px;
                background: white;
                border-radius: 16px;
            }

            button, .download {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 18px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }

            button:disabled {
                opacity: 0.5;
                cursor: wait;
            }

            input {
                display: block;
                margin: 20px 0;
            }

            .images {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 24px;
                margin-top: 24px;
            }

            img {
                max-width: 100%;
                max-height: 500px;
                object-fit: contain;
                border-radius: 8px;
            }

            #result {
                background-color: white;
                background-image: conic-gradient(#ddd 25%, white 0 50%, #ddd 0 75%, white 0);
                background-size: 20px 20px;
            }

            #status {
                margin-top: 16px;
            }

            [hidden] {
                display: none !important;
            }
        </style>
    </head>

    <body>
        <main>
            <h1>Cắt người trong ảnh</h1>
            <p>
                Chọn ảnh JPG hoặc PNG, tối đa 5 MB.
                Tách nền theo đường viền cơ thể, xuất PNG trong suốt.
                Nếu có nhiều người, chọn người có khung lớn nhất.
            </p>

            <form id="upload-form">
                <input
                    id="file"
                    type="file"
                    accept=".jpg,.png"
                    required
                >
                <button id="submit" type="submit">Cắt người</button>
            </form>

            <p id="status" role="status" aria-live="polite"></p>

            <div class="images">
                <section id="original-section" hidden>
                    <h2>Ảnh gốc</h2>
                    <img id="original" alt="Ảnh đã chọn">
                </section>

                <section id="result-section" hidden>
                    <h2>Kết quả</h2>
                    <img id="result" alt="Người được cắt từ ảnh">
                    <p>
                        <a
                            id="download"
                            class="download"
                            download="person.png"
                        >Tải ảnh</a>
                    </p>
                </section>
            </div>

            <p><a href="/docs">Mở Swagger</a></p>
        </main>

        <script>
            const form = document.getElementById("upload-form");
            const fileInput = document.getElementById("file");
            const button = document.getElementById("submit");
            const statusText = document.getElementById("status");
            const original = document.getElementById("original");
            const result = document.getElementById("result");
            const originalSection = document.getElementById("original-section");
            const resultSection = document.getElementById("result-section");
            const download = document.getElementById("download");

            let originalUrl = null;
            let resultUrl = null;

            function clearResult() {
                resultSection.hidden = true;
                result.removeAttribute("src");
                download.removeAttribute("href");

                if (resultUrl) {
                    URL.revokeObjectURL(resultUrl);
                    resultUrl = null;
                }
            }

            fileInput.addEventListener("change", () => {
                clearResult();
                statusText.textContent = "";
                originalSection.hidden = true;
                original.removeAttribute("src");

                if (originalUrl) {
                    URL.revokeObjectURL(originalUrl);
                    originalUrl = null;
                }

                const file = fileInput.files[0];

                if (file) {
                    originalUrl = URL.createObjectURL(file);
                    original.src = originalUrl;
                    originalSection.hidden = false;
                }
            });

            form.addEventListener("submit", async (event) => {
                event.preventDefault();

                const file = fileInput.files[0];
                if (!file) return;

                clearResult();

                if (!/\.(jpg|png)$/i.test(file.name)) {
                    statusText.textContent = "Chỉ chấp nhận file .jpg hoặc .png.";
                    return;
                }

                if (file.size > 5 * 1024 * 1024) {
                    statusText.textContent = "Ảnh không được lớn hơn 5 MB.";
                    return;
                }

                const data = new FormData();
                data.append("file", file);

                button.disabled = true;
                fileInput.disabled = true;
                statusText.textContent = "Đang nhận diện và cắt người...";

                try {
                    const response = await fetch("/crop-person", {
                        method: "POST",
                        body: data,
                    });

                    if (!response.ok) {
                        const error = await response.json().catch(() => ({detail: "Lỗi HTTP " + response.status}));
                        throw new Error(
                            typeof error.detail === "string"
                                ? error.detail
                                : "Không thể xử lý ảnh."
                        );
                    }

                    const blob = await response.blob();
                    resultUrl = URL.createObjectURL(blob);

                    result.src = resultUrl;
                    download.href = resultUrl;
                    resultSection.hidden = false;
                    statusText.textContent = "Đã cắt người thành công!";
                } catch (error) {
                    statusText.textContent = error.message;
                } finally {
                    button.disabled = false;
                    fileInput.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """)
