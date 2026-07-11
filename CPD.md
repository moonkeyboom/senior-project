# CPD Project — Context & Working Notes

> ไฟล์นี้เป็น "หน่วยความจำ" ของโปรเจกต์ สรุปเป้าหมาย งานที่มี และการตัดสินใจสำคัญ
> เพื่อให้ทุก session (และทีม) หยิบงานต่อได้โดยไม่ต้องเริ่มใหม่
> อัปเดตล่าสุด: 2026-07-11

---

## 1. เป้าหมายของโปรเจกต์ (Goal)

ทำ **optimal 1-D k-means** โดยเปลี่ยน objective function จาก withinss (sum of
squared within-cluster distances) มาเป็น **Ω′ (Omega prime)** ซึ่งเป็น metric วัด
"conditional unbiasedness" ของการแบ่งกลุ่ม (grading / CPD — Continuous Performance
Discretization) แทน

พูดง่ายๆ: แทนที่จะหา partition ที่ให้ SSE ต่ำสุด เราต้องการหา partition ของคะแนน
(1-D) ที่ทำให้ **Ω′ สูงสุด**

---

## 2. แหล่งข้อมูลในโปรเจกต์ (Sources)

### 2.1 `Ckmeans.1d.dp_...pdf` — Wang & Song (2011), R Journal
Optimal 1-D k-means ด้วย dynamic programming
- k-means ทั่วไป (Lloyd) ไม่รับประกัน optimal เพราะขึ้นกับ initial centers
- เสนอ DP ที่รับประกัน optimal สำหรับข้อมูล 1 มิติ objective = **withinss**
- Recurrence: `D[i,m] = min over j { D[j-1, m-1] + d(x_j..x_i) }`
  โดย `d()` = sum of squared distances ของ cluster หนึ่ง คำนวณ incremental ผ่าน
  running mean → runtime **O(n²k)**, space O(nk), backtrack ด้วย matrix B
- **จุดสำคัญ**: objective withinss **decompose ได้เป็นผลรวมของแต่ละ cluster
  อิสระต่อกัน** → นี่คือเงื่อนไขที่ทำให้ DP ทำงานได้

### 2.2 `peerjcs2804 1.pdf` — Banditwattanawong & Masdisornchote (2025), PeerJ CS
Norm-referenced CPD + metric **Ω′** (DOI 10.7717/peerj-cs.2804)
- **Ω′ = Ω1 × Ω2 × Ω3** (สมการที่ 2) — ช่วง [0,1] ยิ่งสูงยิ่ง unbiased
  - **Ω1** = 1 − |θ/Θ|  ถ้า Θ ≥ 1, ไม่งั้น = 1
    (θ = จำนวน PRL ที่ไม่ถูก assign, Θ = จำนวน gap ที่กว้างกว่า PVI ที่กว้างสุด —
    Requirement 1: gap ที่กว้างเกิน PVI ต้อง "กัน" ไม่ให้ assign PRL)
  - **Ω2** = (Σδᵢ − ΣD_min) / (ΣD_max − ΣD_min)  ถ้า N ≥ 3, ไม่งั้น = 1
    (δᵢ = gap ระหว่าง lower bound ของ PRL ตัวที่ i กับ upper bound ของตัวที่ i+1;
    D_min/D_max = gap แคบสุด/กว้างสุดลำดับที่ i ในข้อมูลทั้งหมด —
    Requirement 2: maximize gap ระหว่างกลุ่ม)
  - **Ω3** = 1/(1+σ)  ถ้า N ≥ 2, ไม่งั้น = 1
    (σ = std ของความกว้าง PVI ของทุกกลุ่ม — Requirement 3: กลุ่มควรกว้างเท่ากัน)
- N = จำนวน unique PRL (เกรด) ที่ถูก assign จริง
- PVI = Performance Value Interval = max − min ของคะแนนในกลุ่มเดียวกัน
- วิธีที่เปเปอร์เปรียบเทียบ: WGF-CPD, K-CPD (วน k-means หลายรอบ เลือก Ω′ สูงสุด),
  PAM-CPD, Z-score (baseline)
- Algorithm 1 (WGF-CPD): ค่อยๆ ลด N ถ้ายังมี gap > PVI ที่กว้างสุด, วนจน Θ=0

### 2.3 `run_cpd_theta_version.py` — โค้ดของผู้ใช้ (มีอยู่แล้ว)
- `calculate_omega()` — คำนวณ Ω เวอร์ชันย่อ **ไม่ตรงกับ Ω′ เต็มสมการในเปเปอร์เป๊ะ**
  (รวม σ กับ delta-gap เข้าด้วยกันเป็นสูตรเดียว ไม่ได้แยก Ω1·Ω2·Ω3 ชัดเจน —
  ⚠️ ควรตรวจสอบ/จัดให้ตรงเปเปอร์ถ้าต้องการ Ω′ จริง)
- `grading_by_heuristic()` — หา partition ด้วยการ **brute-force enumerate ทุก
  combination ของจุดตัด candidate** (เลือก candidate จาก gap ที่กว้างสุดก่อน) แล้ว
  เลือกอันที่ Ω สูงสุด → **ไม่ optimal เต็มรูป** เพราะจำกัด candidate pool ไว้ก่อน
- `cpd_refinement_old_loop` vs `cpd_refinement_theta_loop` — เปรียบเทียบ 2 loop
  ตาม Algorithm 1: old = `while iter < max_iter` (ผิดตามเปเปอร์),
  theta = `while j <= θ` (ถูกตามเปเปอร์) โดยลดเกรดถ้า max_gap > max_PVI
- อินพุตในโค้ด: `./file/input/221.xlsx`, เกรด `['A','B+','B','C+','C','D+','D','F']`

### 2.4 `ggสเปรดชีตไม่มีชื่อ  WGP.csv` — ข้อมูล
⚠️ เป็น blob ที่ระบบดึงข้อความไม่ได้ — **ยังไม่เห็นเนื้อหาจริง** ต้องให้ผู้ใช้
อัปโหลดใหม่เป็น attachment หรือให้ path ถึงจะใช้งานได้

---

## 3. Insight สำคัญ (ต้องตัดสินใจก่อนเขียนโค้ด)

DP ของ Ckmeans.1d.dp ทำงานได้เพราะ withinss **decompose เป็นผลรวมของแต่ละ cluster
แบบอิสระ** แต่ **Ω′ ไม่ decompose แบบนั้น**:
- Ω1 ต้องรู้ธรณีประตู "gap ที่กว้างที่สุด" ของทั้งชุด
- Ω2 ต้องรู้อันดับ gap แคบสุด/กว้างสุดในข้อมูลทั้งหมด (global)
- Ω3 ต้องรู้ σ ของความกว้างทุกกลุ่มพร้อมกัน

→ DP recurrence มาตรฐานใช้ตรงๆ กับ Ω′ ไม่ได้ ต้องเลือกกลยุทธ์:

| แนวทาง | รับประกัน optimal ของ Ω′? | Runtime | เหมาะกับ |
|--------|--------------------------|---------|----------|
| **Exhaustive** (ลองทุกจุดตัด C(n-1,k-1)) | ใช่ | แพง | n,k เล็ก (n<100, k<10) |
| **DP หา SSE ก่อน แล้วปรับ Ω′** | ไม่ (heuristic) | O(n²k) | n ใหญ่ |
| **ทำทั้งสองแล้ว benchmark** | — | — | ตรวจว่าเท่ากันไหม |

---

## 4. คำถามที่ยังค้าง (Open decisions)

1. **Search method**: exhaustive (optimal จริง) vs DP-heuristic vs ทำทั้งสองเทียบกัน
2. **k handling**: fixed k (เช่น 8 เกรด) vs ลอง k=2..k_max แล้วเลือก k ที่ Ω′ สูงสุด
3. **Data source**: ใช้ WGP.csv (ต้องอัปโหลดใหม่) / อัปโหลดไฟล์อื่น / สร้าง mock data
4. **Ω vs Ω′**: จะจัด `calculate_omega()` ให้ตรงสมการ Ω′=Ω1·Ω2·Ω3 ในเปเปอร์ไหม

---

## 5. สถานะ (Status)
- [x] อ่านและสรุปเปเปอร์ทั้ง 2 + โค้ดผู้ใช้
- [ ] ยืนยันแนวทาง (ข้อ 4 ด้านบน)
- [ ] เข้าถึงข้อมูลจริง (WGP.csv)
- [ ] Implement optimal-Ω′ partitioner
- [ ] Verify / benchmark