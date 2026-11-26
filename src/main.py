import cv2
from ultralytics import YOLO


def deteksi_gelap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness < 50:
        status = "GELAP"
    else:
        status = "NORMAL"

    return brightness, status


def main():
    # Memuat model YOLO pretrained
    model = YOLO("yolov8n.pt")

    # Membuka webcam
    kamera = cv2.VideoCapture(0)

    if not kamera.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    print("Kamera berhasil terhubung.")
    print("Deteksi manusia dan tracking dimulai.")
    print("Tekan Q untuk keluar.")

    while True:
        berhasil, frame = kamera.read()

        if not berhasil:
            print("Gagal membaca gambar dari kamera.")
            break

        # Deteksi manusia dan tracking menggunakan ByteTrack
        hasil = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Menampilkan hasil deteksi dan ID tracking
        frame_deteksi = hasil[0].plot()

        # Mendeteksi kecerahan
        brightness, status_gelap = deteksi_gelap(frame)

        # Menampilkan informasi kecerahan
        cv2.putText(
            frame_deteksi,
            f"Kecerahan: {brightness:.2f} ({status_gelap})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Menampilkan hasil
        cv2.imshow("Deteksi Aktivitas Area Kampus", frame_deteksi)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    kamera.release()
    cv2.destroyAllWindows()

    print("Sistem deteksi aktivitas dihentikan.")


if __name__ == "__main__":
    main()