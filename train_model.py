# train_autoencoder_clean_only.py
# تدريب Autoencoder على برامج سليمة فقط + توليد نسخ (augmentation) + دالة كشف
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # مهم: قبل استيراد tensorflow عشان يخفي الرسائل

import random
import numpy as np
from PIL import Image
import shutil
import json

# التحذيرات والـ logs
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

# ML imports (بعد ضبط env)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------- إعداد المسارات والإعدادات -------
CLEAN_EXE_DIR = r"D:\file\ExeVisionAI\data\raw_exe\benign"      # ← **مسارك المحدث**
TEMP_IMG_DIR = "tmp_images"              # صور مؤقته من الـ exe
AUG_IMG_DIR = "tmp_images_aug"           # صور بعد الـ augmentation
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_clean.h5")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "recon_threshold.json")

IMG_SIZE = (128, 128)   # حجم الصورة للتدريب (أمن للذاكرة). عدّله لو عندك GPU وذاكرة.
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 30
RANDOM_SEED = 42

os.makedirs(TEMP_IMG_DIR, exist_ok=True)
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------- دالة تحويل EXE إلى صورة --------
def exe_to_image(exe_path, image_path, size=IMG_SIZE):
    """
    تحول ملف exe إلى صورة RGB ثابتة الحجم.
    طريقة بسيطة: قراءة bytes ووضعها في مصفوفة.
    """
    try:
        with open(exe_path, "rb") as f:
            content = f.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        length = size[0] * size[1] * CHANNELS
        if arr.size < length:
            arr = np.pad(arr, (0, length - arr.size), mode='constant', constant_values=0)
        else:
            arr = arr[:length]
        arr = arr.reshape((size[0], size[1], CHANNELS))
        img = Image.fromarray(arr)
        img.save(image_path)
        return True
    except Exception as e:
        print(f"[!] Failed to convert {exe_path} -> {e}")
        return False

# ------- تجهيز صور نظيفة من ملفات exe -------
def prepare_images_from_exes(clean_dir=CLEAN_EXE_DIR, out_dir=TEMP_IMG_DIR):
    print("[*] Converting EXEs to images...")
    paths = []
    for fname in os.listdir(clean_dir):
        if not fname.lower().endswith(".exe"):
            continue
        src = os.path.join(clean_dir, fname)
        dst = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.png")
        ok = exe_to_image(src, dst)
        if ok:
            paths.append(dst)
    print(f"[+] Converted {len(paths)} EXE files to images.")
    return paths

# ------- عمل Data Augmentation (توليد نسخ متحركة للصور) -------
def augment_images(input_paths, out_dir=AUG_IMG_DIR, copies_per_image=5):
    """
    يستخدم ImageDataGenerator لعمل تعديلات على الصور لزيادة التنوع.
    لا نغير البايتات الأصلية للـ EXE هنا؛ نعدل الصور الناتجة.
    """
    print("[*] Augmenting images...")
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    saved = []
    for path in input_paths:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        x = np.array(img) / 255.0
        x = x.reshape((1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
        # نحفظ الصورة الأصلية في المخرج أيضًا
        base_name = os.path.splitext(os.path.basename(path))[0]
        orig_out = os.path.join(out_dir, f"{base_name}_orig.png")
        Image.fromarray((x[0]*255).astype(np.uint8)).save(orig_out)
        saved.append(orig_out)
        # توليد نسخ augment
        gen = datagen.flow(x, batch_size=1)
        for i in range(copies_per_image):
            batch = next(gen)[0]
            out_name = os.path.join(out_dir, f"{base_name}_aug{i}.png")
            Image.fromarray((batch*255).astype(np.uint8)).save(out_name)
            saved.append(out_name)
    print(f"[+] Augmented and saved {len(saved)} images to {out_dir}")
    return saved

# ------- تحميل الصور كمصفوفة numpy للتدريب -------
def load_image_paths_to_array(paths):
    arr = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            arr.append(np.array(img) / 255.0)
        except:
            continue
    if not arr:
        return np.empty((0, IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
    return np.stack(arr, axis=0).astype('float32')

# ------- بناء Autoencoder بسيط (CNN) -------
def build_autoencoder(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS), latent_dim=128):
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)  # 64x64
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)  # 32x32
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)  # 16x16

    # code
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    x = layers.Dense(16*16*128, activation='relu')(encoded)
    x = layers.Reshape((16,16,128))(x)
    x = layers.UpSampling2D((2,2))(x)  # 32x32
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)  # 64x64
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)  # 128x128
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    decoded = layers.Conv2D(CHANNELS, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# ------- احتساب خطأ إعادة الإعمار (MSE) -------
def reconstruction_errors(model, X):
    recon = model.predict(X, verbose=0)
    errors = np.mean(np.square(recon - X), axis=(1,2,3))
    return errors

# ------- دالة تدريب شاملة -------
def train_pipeline():
    # 1) تحويل EXE -> صور
    paths = prepare_images_from_exes(CLEAN_EXE_DIR, TEMP_IMG_DIR)
    if not paths:
        print("[!] No EXE files found in clean directory. Exiting.")
        return

    # 2) عمل augmentation وحفظها
    aug_paths = augment_images(paths, AUG_IMG_DIR, copies_per_image=5)  # عدّل العدد لو عايز

    # 3) تحميل كل الصور كمصفوفة للتدريب
    X = load_image_paths_to_array(aug_paths)
    print(f"[+] Training data shape: {X.shape}")

    # 4) بناء الموديل
    autoencoder = build_autoencoder()
    autoencoder.summary()

    # 5) تدريب
    print("[*] Training autoencoder...")
    autoencoder.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, shuffle=True)

    # 6) احتساب خطأ الإعادة على مجموعة التدريب لتحديد threshold
    errors = reconstruction_errors(autoencoder, X)
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    # نضع threshold = mean + 3*std (قيمة افتراضية يمكن تعديلها)
    threshold = mean_err + 3 * std_err
    print(f"[+] Reconstruction error mean: {mean_err:.6f}, std: {std_err:.6f}")
    print(f"[+] Using threshold = {threshold:.6f}")

    # 7) حفظ الموديل والعتبة
    autoencoder.save(MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "mean": mean_err, "std": std_err}, f)
    print(f"[✓] Model saved to {MODEL_PATH}")
    print(f"[✓] Threshold saved to {THRESHOLD_PATH}")

# ------- دالة كشف ملف جديد -------
def detect(exe_path, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
    if not os.path.exists(exe_path):
        print(f"[✘] File not found: {exe_path}")
        return

    # تحويل exe إلى صورة مؤقتة
    tmp = "detect_tmp.png"
    ok = exe_to_image(exe_path, tmp)
    if not ok:
        print("[!] Failed to convert exe for detection.")
        return

    # تحميل الموديل والعتبة
    model = tf.keras.models.load_model(model_path)
    with open(threshold_path, "r") as f:
        info = json.load(f)
    threshold = float(info["threshold"])

    # تحضير الصورة
    img = Image.open(tmp).convert("RGB").resize(IMG_SIZE)
    X = np.expand_dims(np.array(img) / 255.0, axis=0).astype('float32')

    # حساب خطأ إعادة الإعمار
    err = reconstruction_errors(model, X)[0]
    os.remove(tmp)

    # قرار
    # lower error => مشابه للسليم. higher error => مشتبه
    is_clean = err <= threshold
    confidence = max(0.0, 1.0 - (err - info["mean"]) / (info["std"]*6 + 1e-9))  # تقدير تقريبي للثقة
    confidence = float(np.clip(confidence, 0.0, 1.0))

    label = "سليم" if is_clean else "مشتبه"
    print(f"[⚠️] {exe_path} → {label} (recon_err={err:.6f}, threshold={threshold:.6f}, confidence={confidence:.2f})")
    return {"path": exe_path, "label": label, "recon_err": float(err), "threshold": threshold, "confidence": confidence}

# ------- Main runner -------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train autoencoder on clean EXEs and detect anomalies.")
    parser.add_argument("--train", action="store_true", help="Run training pipeline (convert, augment, train, save model).")
    parser.add_argument("--detect", type=str, help="Detect single exe file path (requires trained model).")
    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.detect:
        detect(args.detect)
    else:
        print("Usage examples:")
        print("  python train_autoencoder_clean_only.py --train")
        print("  python train_autoencoder_clean_only.py --detect D:/path/to/file.exe")
