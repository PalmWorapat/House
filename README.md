# 🏠 อธิบายโจทย์: House Recognition - Image Classification

**เป้าหมายหลัก (Task):** โจทย์นี้เป็นปัญหาการจำแนกประเภทภาพแบบสองคลาส (Binary Classification) โดยโมเดลจะต้องตรวจจับและแยกแยะว่าภาพถ่ายมุมมองถนน (Street-view image) ที่กำหนดให้ มองเห็น "บ้าน" อยู่ในภาพหรือไม่ (มีบ้าน = Yes/1, ไม่มีบ้าน = No/0)

**รายละเอียดชุดข้อมูล (Dataset):**
* **Training Set:** ภาพสำหรับฝึกสอนจำนวน 2,954 ภาพ
* **Test Set:** ภาพสำหรับทดสอบจำนวน 1,550 ภาพ

**รูปแบบการส่งคำตอบ (Submission Format):**
ผู้เข้าร่วมต้องส่งไฟล์ในรูปแบบ `sample_submission.csv` โดยประกอบด้วย 2 คอลัมน์หลัก คือ:
1.  `id`: รหัสหรือชื่อไฟล์ภาพ
2.  `answer`: ผลการทำนาย (0 หรือ 1)

**เกณฑ์การประเมินผล (Evaluation Metric):**
* ประเมินผลด้วยค่าความแม่นยำ (Accuracy Score)
* การคิดคะแนนแบ่งเป็น Public Leaderboard 50% และ Private Leaderboard 50%

---

# 💻 อธิบาย Pipeline Code

จากไฟล์ Jupyter Notebook (`house_best.ipynb`) การทำงานของโค้ดถูกออกแบบมาอย่างเป็นระบบตามมาตรฐานของงาน Computer Vision โดยมีขั้นตอนดังนี้:

### 1. การติดตั้งและนำเข้าไลบรารี (Install & Import Libraries)
* **การติดตั้ง:** โค้ดเริ่มต้นด้วยการติดตั้งไลบรารีที่จำเป็น ได้แก่ `timm` (สำหรับเรียกใช้โมเดล PyTorch ที่ pre-trained มาแล้ว), `albumentations` (สำหรับการทำ Image Augmentation) และ `kaggle` (สำหรับจัดการข้อมูล)
* **การนำเข้า:** มีการเรียกใช้ PyTorch, Pandas, Numpy, Matplotlib, Seaborn และเครื่องมือประเมินผลจาก Scikit-learn (เช่น StratifiedKFold, accuracy_score, roc_auc_score) รวมถึงการตรวจสอบการใช้งาน GPU (CUDA)

### 2. การจัดการชุดข้อมูล (Download & Extract Dataset)
* ใช้ Kaggle API เพื่อดาวน์โหลดชุดข้อมูลการแข่งขันลงมาในโฟลเดอร์ `data`
* ทำการแตกไฟล์ `.zip` ทั้งหมดโดยอัตโนมัติ เพื่อให้ได้โฟลเดอร์ภาพ train/test และไฟล์ CSV ที่เกี่ยวข้อง

### 3. การสำรวจและเตรียมข้อมูลเบื้องต้น (Exploratory Data Analysis - EDA)
* โหลดไฟล์ `train.csv` และ `sample_submission.csv`
* **การจัดการข้อมูล:** ทำการเปลี่ยนชื่อคอลัมน์ให้เข้าใจง่ายขึ้นและสอดคล้องกันทั้งโปรเจกต์ (เปลี่ยน `image_name` เป็น `id` และเปลี่ยน `class` เป็น `answer`)
* **วิเคราะห์ความสมดุล:** ตรวจสอบการกระจายตัวของคลาสในชุดข้อมูลฝึกสอน (Class Distribution) โดยพบว่าอัตราส่วนระหว่างภาพที่มีบ้านและไม่มีบ้านมีความสมดุลกันในระดับที่ดี (อัตราส่วนประมาณ 0.943)

### 4. การสร้างโมเดล (Model Configuration)
* **สถาปัตยกรรม (Architecture):** โค้ดเลือกใช้โมเดล **EfficientNet-B3** ซึ่งเป็นโมเดลที่มีประสิทธิภาพสูง
* **Pre-trained Weights:** ใช้การเรียนรู้ถ่ายโอน (Transfer Learning) จากน้ำหนักของโมเดลที่ถูกฝึกสอนมาแล้วด้วยชุดข้อมูล ImageNet
* **Custom Head:** มีการปรับแต่งส่วนหัวของโมเดล (Fine-tuning with custom head) เพื่อให้ส่งออกผลลัพธ์เป็น 2 คลาสตามที่โจทย์ต้องการ

### 5. การตั้งค่าการฝึกสอน (Training Setup & Execution)
ถึงแม้โค้ดส่วน Training Loop จะไม่ได้แสดงใน EDA ทั้งหมด แต่จากส่วนสรุปผลลัพธ์ (Final Summary) แสดงให้เห็นถึง Pipeline การเทรนที่รัดกุม:
* **Validation Strategy:** มีการใช้ Cross Validation Folds (`n_folds`) เพื่อป้องกันการ Overfitting และประเมินผลโมเดลได้แม่นยำขึ้น
* **Hyperparameters:** มีการปรับแต่ง Epochs, Batch size, ตัวเพิ่มประสิทธิภาพ (Optimizer) และ Learning Rate Scheduler อย่างเป็นระบบ

### 6. การประเมินผลและการสร้างไฟล์ส่งคำตอบ (Evaluation & Submission)
* **OOF Evaluation:** โค้ดมีการคำนวณและแสดงค่า OOF Accuracy (Out-of-Fold Accuracy) และ OOF AUC-ROC เพื่อดูประสิทธิภาพโดยรวม
* **Prediction:** นำโมเดลไปทำนายภาพใน Test set เพื่อตรวจดูสัดส่วนการทำนาย (เช่น จำนวนภาพที่ทำนายว่าเป็นบ้านและไม่เป็นบ้าน)
* **Submission:** บันทึกผลลัพธ์สุดท้ายลงในไฟล์ `submission.csv` พร้อมที่จะนำไปอัปโหลดส่งในระบบทันที
