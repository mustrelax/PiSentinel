# 30 Kasım 2025 Pazartesi - Çalışma: Drone Takip Sistemi Optimizasyonu ve Stabilite

Bugün PiSentinel projesi üzerinde oldukça kapsamlı bir çalışma gerçekleştirdim. Ana hedefim, sistemin Raspberry Pi 5 üzerindeki performansını artırmak ve özellikle uzak mesafedeki küçük droneları daha kararlı bir şekilde takip edebilmesini sağlamaktı.

Güne **ROI (İlgi Alanı) Takibi** ve **Adaptive Inference** özelliklerini devreye alarak başladım. Bu sayede sistem, hedefe kilitlendiğinde tüm ekranı işlemek yerine sadece ilgili bölgeye odaklanarak hem işlem gücünden tasarruf etti hem de tepki süresini iyileştirdi.

Arayüz tarafında, **HUD (Heads-Up Display)** tasarımını tamamen elden geçirdim. Daha minimal ve profesyonel bir görünüme geçiş yaptım. Kafa karışıklığını önlemek adına FPS yerine **Inference Time (ms)** gösterimine geçtim ve kritik bilgileri sağ alt köşeye taşıyarak görüş alanını ferahlattım. Ancak henüz planladığım gibi çalışmıyor. Sadece 5% kadar iyileştirme yaptığını söyleyebilirim.

Karşılaştığım en büyük zorluklardan biri, droneların yeşillik veya dağ manzarası gibi karmaşık arka planlarda kamufle olmasıydı. Bunu çözmek için **Gamma Düzeltmesi** ve **Dinamik Güven Eşiği** (Dynamic Confidence Threshold) mekanizmalarını ekledim. Artık sistem bir hedefe kilitlendiğinde, onu kaybetmemek için çok daha hassas davranıyor ve gölgeli alanlardaki detayları daha iyi analiz ediyor.

Uzak mesafedeki çok küçük (5-10px) droneların ilk tespiti (initial detection) sorununu çözmek için **"Sektör Taramalı Zoom" (Sector Scanning)** adını verdiğim yaratıcı bir yöntem geliştirdim. Sistem artık boşta iken ekranı sadece merkezden değil, sırasıyla köşelerden de "zoom" yaparak tarıyor. Bu sayede kenarlardan giren uzak tehditleri çok daha erken yakalayabiliyorum.

Günün son ve en kritik dokunuşu ise **"Spatial Lock" (Konumsal Kilitleme)** algoritması oldu. Sistemin kalabalık sürüler ile tekli hedefler arasında kararsız kalıp sürekli hedef değiştirmesi (zıplama) sorununu bu sayede çözdüm. Artık takip edilen hedefin ID'si değişse veya anlık olarak kaybolsa bile, sistem hemen uzaktaki başka bir hedefe atlamak yerine, son görülen konumdaki "mirasçı" hedefi arıyor ve takibi bırakmıyor.

Sonuç olarak; hem görsel açıdan daha profesyonel, hem de algoritmik olarak çok daha zeki, kararlı ve zorlu koşullara dayanıklı bir sisteme ulaştım.

# 1 Aralık 2025 Pazartesi - Çalışma: Stabilizasyon, Sentry Modu ve Hassasiyet Dengesi

Bugün sistemin stabilizasyonu ve "tek drone" senaryosundaki performansı üzerine yoğunlaştım. Dün yapılan geliştirmelerin ardından ortaya çıkan bazı yan etkileri (salınım, yanlış tespitler) gidermek için bir dizi iyileştirme yaptım.

İlk olarak, **Servo Kontrol Algoritmasını** tamamen elden geçirdim. Kameranın hedefe kilitlendiğinde sürekli yukarı-aşağı titremesi (oscillation) sorununu çözmek için PID katsayılarını yumuşattım ve "Dead Zone" (Ölü Bölge) yarıçapını artırdım. Artık kamera, hedef merkeze yakınken gereksiz mikro hareketler yapmıyor, bu da görüntünün çok daha stabil olmasını sağlıyor.

İkinci önemli geliştirme, **"Gözcü (Sentry) Modu"** oldu. Sistem boşta kaldığında (hedef bulamadığında) artık sabit durmak yerine, tıpkı bir güvenlik kamerası gibi sağa-sola yumuşak bir sinüs dalgası hareketiyle (Sine Wave) tarama yapıyor. Bu moda özel olarak **"Akıllı Zoom"** özelliğini entegre ettim; yani kamera sadece boşta ve tarama yaparken zoom yaparak uzaktaki tehditleri arıyor, ancak bir hedefe kilitlendiğinde veya yerel arama yaparken zoom yapmayarak kafa karışıklığını önlüyor.

Günün en zorlu kısmı, **Hassasiyet vs. Gürültü (Sensitivity vs. Noise)** dengesini kurmaktı. P2 modelinin yüksek hassasiyeti, çimen ve taş gibi dokuların da drone sanılmasına yol açıyordu. Bunu çözmek için **"Süreklilik Filtresi" (Persistence Filter)** geliştirdim. Artık bir nesnenin hedef kabul edilmesi için sadece anlık görünmesi yetmiyor, belirli bir süre (4-6 kare) boyunca istikrarlı bir şekilde takip edilmesi gerekiyor. Bu sayede titreyen gürültüleri (taş, çimen) başarıyla elerken, uzaktaki gerçek droneları kaçırmamayı başardım.

Şu anki haliyle sistem, tekli drone takibinde (1vs1) oldukça kararlı, "Tam Güdümlü" bir his veriyor ve uzağı görebiliyor.

**Yarının Planı:**
Yarın odak noktamı **Sürü (Swarm) ve Çoklu Drone** senaryolarına çevireceğim. Tekli takibi bozmadan, birden fazla hedefin olduğu durumlarda sistemin nasıl davranacağını dengelemeye çalışacağım. Ondan sonra sistemi olduğu gibi kabul edip diğer özelliklerin implementasyonuna geçeceğim.

---

# 2 Aralık 2025 Salı - Araştırma: Sürü Senaryoları ve Dinamik Tehdit Değerlendirmesi

Bugün sistemin en büyük eksikliğine odaklandım: **Çoklu drone (swarm) senaryolarında hangi hedefe kilitlenmeli?** Şu an tekil drone takibinde oldukça başarılıyım ama sürü karşısında veya birden fazla hedefin olduğu durumda sistem ilk gördüğüne kilitlenip hiç değiştirmiyor. Bunun için askeri sistemlerin yaklaşımlarını araştırdım ve kapsamlı bir literatür taraması yaptım.

Mevcut sistemdeki sorun basit ama kritik: PRIMARY LOCK mekanizması bir hedefe kilitlendiğinde statik 1000 puanlık bir bonus veriyor ve bu yüzden daha tehlikeli bir hedef ortaya çıksa bile asla geçiş yapamıyor. Hatta hedef kaybolsa bile son pozisyona takılı kalabiliyor.

**Askeri C-RAM/CIWS Sistemlerinden Öğrendiklerim:**

Lockheed Martin C-UAS sistemleri ve CNAS raporlarını inceleyerek öğrendiğim en önemli şey, askeri sistemlerin **Time-To-Impact (TTI)** metriğini öncelik olarak kullanması. Yani hedefin merkeze (veya korunacak alana) ulaşma süresini hesaplayıp, en kısa sürede gelecek olan hedefe öncelik veriyorlar. Bu çok mantıklı çünkü uzaktaki yavaş hedeften çok, hızla yaklaşan yakın hedef daha tehlikelidir.

İkinci öğrendiğim şey ise **Three-Way Decision Framework** yaklaşımı. Hedefleri üç zona ayırıyorlar:
- RED (Kritik): TTI < 5 saniye veya mesafe < 150 piksel
- YELLOW (Orta): TTI 5-15 saniye arası veya mesafe 150-300 piksel
- GREEN (Düşük): Daha uzak veya yavaş hedefler

Her zonun farklı bir "kilit gücü" var. Kırmızı zonda daha güçlü kilit, yeşilde daha zayıf. Bu sayede geçiş kararları daha akıllıca veriliyor.

**Sürü Lider Tespiti Doktrini:**

Askeri sistemlerde "Kill the leader, swarm disperses" prensibi var. Sürü formasyonunda lideri tespit edip onu vurduğunda, geri kalanı dağılıyor. Lider tespiti için üç faktör kullanılıyor:
- En önde olan (merkeze en yakın): Ağırlık %50
- En merkezi olan (oluşum merkezine yakın): Ağırlık %20
- En hızlı yaklaşan: Ağırlık %30

**Dinamik Yeniden Değerlendirme:**

En kritik bulgu ise periyodik olarak tüm hedefleri yeniden değerlendirme mekanizması. Kilitli bile olsan 2-3 saniyede bir tüm alternatifleri kontrol edip, daha tehlikeli bir hedef varsa ona geçiyorsun. Ancak bu geçişler zonlara göre farklı eşiklerle yapılıyor:
- Daha yüksek tehdide geçiş (YELLOW'dan RED'e): Kolay, sadece 50 puan fark yeterli
- Aynı zon içinde geçiş: Orta zorluk, 30-80 puan fark
- Daha düşük tehdide geçiş: Zor, 150 puan fark gerekli

Bu sayede sistemi gereksiz zıplamalardan korurken, gerçekten kritik durumlarda hızlı reaksiyon verebilmesini sağlıyorlar.

**Akademik Kaynaklar (Tez İçin):**

Bu araştırma sırasında birkaç önemli kaynak buldum:
- IEEE Multi-Target Threat Assessment makaleleri (2025)
- AIAA Weapon-Target Allocation Studies
- Time-To-Impact Based Priority Selection araştırmaları
- Swarm Defense Architecture çalışmaları

**Bugünün Sonucu:**

Detaylı bir implementasyon planı hazırladım ama henüz koda dökmedim. Plan şu adımları içeriyor:
1. TTI hesaplama motoru
2. Üç katmanlı tehdit zonları (RED/YELLOW/GREEN)
3. Adaptif kilit bonusu sistemi (50-200 puan, statik 1000 yerine)
4. Senaryo tespiti (SINGLE/SMALL_GROUP/MEDIUM_SWARM/LARGE_SWARM)
5. Sürü lider tespit algoritması
6. Dinamik re-assessment timer (2-3 saniye interval)

Sistem teorik olarak şu anda kullandığım basit PRIMARY LOCK'tan çok daha akıllı olacak. Dakikada 0.1 geçiş yerine 2-4 geçiş yapabilecek ve bunlar mantıklı, açıklanabilir geçişler olacak.

**Neden Henüz Uygulamadım:**

Bu kadar büyük bir değişiklik riskli. Şu anki tekil takip performansını bozmak istemiyorum. Önce mevcut sistemi daha da stabilize etmem, sonra bu yeni tehdit değerlendirme sistemini aşamalı olarak entegre etmem gerekiyor. Belki feature flag'lerle açıp kapanabilir hale getiririm.

**Yarının Planı:**

Şimdilik bu planı bir kenara not ettim. Önce mevcut sistemdeki küçük sorunları (salınım, hassasiyet dengesi, vb.) tam olarak çözeceğim. Sonra bu dinamik tehdit değerlendirme sistemini test ortamında deneyeceğim.
