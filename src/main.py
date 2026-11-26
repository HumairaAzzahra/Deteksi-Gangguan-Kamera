import cv2
import time
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
    model = YOLO("yolov8n.pt")
    kamera = cv2.VideoCapture(0)

    if not kamera.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    # Menyimpan waktu pertama kali setiap ID terdeteksi
    data_tracking = {}

    print("Kamera berhasil terhubung.")
    print("Deteksi manusia dan pencatatan durasi dimulai.")
    print("Tekan Q untuk keluar.")

    while True:
        berhasil, frame = kamera.read()

        if not berhasil:
            print("Gagal membaca gambar dari kamera.")
            break

        hasil = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )

        frame_deteksi = hasil[0].plot()
        waktu_sekarang = time.time()

        # Mengambil ID objek yang terdeteksi
        boxes = hasil[0].boxes

        if boxes is not None and boxes.id is not None:
            id_objek = boxes.id.int().cpu().tolist()

            for id_person in id_objek:
                # Mencatat waktu pertama ID muncul
                if id_person not in data_tracking:
                    data_tracking[id_person] = waktu_sekarang

                # Menghitung durasi sejak pertama terdeteksi
                durasi = waktu_sekarang - data_tracking[id_person]

                # Mencari posisi bounding box
                for box, id_box in zip(boxes, id_objek):
                    if id_box == id_person:
                        x1, y1, x2, y2 = box.xyxy[0].int().cpu().tolist()

                        # Menampilkan durasi pada bounding box
                        cv2.putText(
                            frame_deteksi,
                            f"Durasi: {durasi:.1f}s",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
                        break

        # Mendeteksi kecerahan
        brightness, status_gelap = deteksi_gelap(frame)

        cv2.putText(
            frame_deteksi,
            f"Kecerahan: {brightness:.2f} ({status_gelap})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Deteksi Aktivitas Area Kampus", frame_deteksi)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    kamera.release()
    cv2.destroyAllWindows()

    print("Sistem deteksi aktivitas dihentikan.")


if __name__ == "__main__":
    main()