from ultralytics import YOLO
# يجب وضع الكود داخل هذا الشرط في بيئة ويندوز لتجنب تجمد الجهاز (Multiprocessing Crash)
if __name__ == '__main__':
    # 1. تحميل المعمارية الأحدث بأوزانها المسبقة
    model = YOLO("yolo11s.pt")

    # 2. بدء التدريب المتقدم (Advanced Training Loop)
    results = model.train(
        # --- الإعدادات الأساسية والمسارات ---
        # ضع المسار الكامل (Absolute Path) لضمان عدم حدوث خطأ FileNotFoundError
        data=r"data.yaml", 
        project="Drone_Vision_Core",
        name="final_training",
        epochs=300,          
        patience=60,         
        
        # إذا واجهت خطأ (CUDA Out of Memory)، قم بخفض الدفعة إلى 8
        batch=12,            
        imgsz=640,
        device=0,
        
        # --- تحسين أداء العتاد (Windows Optimized) ---
        workers=6,           # تم خفضها لـ 4 لضمان استقرار نظام ويندوز
        amp=True,            

        # --- خوارزميات التحسين المتقدمة (Advanced Optimization) ---
        optimizer="AdamW",   
        lr0=0.001,           
        lrf=0.01,            
        weight_decay=0.0005, 
        cos_lr=True,         

        # --- تضخيم البيانات المخصص لطيران الدرون (Drone-Specific Augmentation) ---
        degrees=15.0,        
        scale=0.5,           
        perspective=0.0001,  
        fliplr=0.5,          
        
        # --- التكتيكات العنيفة للعوائق الصعبة (Aggressive Augmentation) ---
        mosaic=1.0,          
        mixup=0.2,           
        copy_paste=0.1,      
        
        # --- اللمسة النهائية (Fine-Tuning Phase) ---
        close_mosaic=15      
    )