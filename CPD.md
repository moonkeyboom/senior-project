# CPD Project — Context & Working Notes

> ไฟล์นี้เป็น "หน่วยความจำ" ของโปรเจกต์ สรุปเป้าหมาย งานที่มี และการตัดสินใจสำคัญ
> เพื่อให้ทุก session (และทีม) หยิบงานต่อได้โดยไม่ต้องเริ่มใหม่
> อัปเดตล่าสุด: 2026-07-11 (เพิ่มผล implementation + findings)

---

## 1. เป้าหมายของโปรเจกต์ (Goal)

ทำ **optimal 1-D k-means** โดยเปลี่ยน objective function จาก withinss (sum of
squared within-cluster distances) มาเป็น **Ω′ (Omega prime)** ซึ่งเป็น metric วัด
"conditional unbiasedness" ของการแบ่งกลุ่ม (grading / CPD — Conditional Performance
Discretization) แทน — คือหา partition ของคะแนน (1-D) ที่ทำให้ **Ω′ สูงสุด**

---

## 2. แหล่งข้อมูลในโปรเจกต์ (Sources)

### 2.1 `Ckmeans.1d.dp_...pdf` — Wang & Song (2011), R Journal
DP ที่รับประกัน optimal 1-D k-means (objective = withinss)
- Recurrence: `D[i,m] = min over j { D[j-1,m-1] + d(x_j..x_i) }`, runtime O(n²k)
- **withinss decompose ได้** เป็นผลรวมของแต่ละ cluster → DP ใช้ได้ตรง

### 2.2 `peerjcs2804 1.pdf` — Banditwattanawong & Masdisornchote (2025), PeerJ CS
เสนอ metric **Ω′** (DOI 10.7717/peerj-cs.2804), Eq. (2): Ω′ = Ω1·Ω2·Ω3
- เสนอ 4 วิธี: WGF-CPD (heuristic widest-gap), K-CPD (k-means วนแล้วเลือก Ω′ สูงสุด),
  PAM-CPD, M-CPD (รวมทั้งหมดเลือกดีสุด)
- Algorithm 1–4: เริ่มจาก |L| เกรด แล้ว "ลด" จำนวนเกรดตาม Requirement 1 (while j ≤ Θ)

### 2.3 `run_cpd_theta_version.py` — โค้ดเดิมของผู้ใช้
- `calculate_omega()` เวอร์ชันย่อ **ไม่ตรง Ω′ เต็มสมการ** (ใช้ np.std = population,
  ไม่แยก Ω1·Ω2·Ω3) → ถูกแทนที่ด้วย `calculate_omega_prime()` ใหม่ (ดูข้อ 6)

### 2.4 `ggสเปรดชีตไม่มีชื่อ WGP.csv` — ข้อมูล
⚠️ เป็น blob อ่านข้อความไม่ได้ — **ยังไม่เห็นข้อมูลจริง** ต้องอัปโหลดใหม่เป็น attachment

---

## 3. ผลลัพธ์ Implementation (ไฟล์ใหม่ในโปรเจกต์)

### `optimal_cpd_omega_prime.py` — โมดูลหลัก
- `calculate_omega_prime(values_desc, cuts, num_labels, U, L, ddof=1)` — Ω′ ตาม Eq.(2)
- `exhaustive_optimal(...)` — brute-force C(n-1,k-1) รับประกัน optimal จริง (fixed-k หรือ sweep k)
- `ckmeans_1d_dp(...)` + `dp_best(...)` — DP O(n²k) หา SSE-optimal ต่อ k แล้วเลือก k ที่ Ω′ สูงสุด

### `run_cpd_omega_prime.py` — verify + benchmark
- รับ path CSV/xlsx ได้ (`python3 run_cpd_omega_prime.py scores.csv`); ไม่ใส่ = ใช้ demo Table 3
- ผ่าน 2 verification: (1) Table 3 → A1=0.0824≈0.08, A2=0.2205≈0.22, และ Ω1/Ω2/σ ตรง;
  (2) exhaustive ≥ DP ทุก k

---

## 6. FINDINGS สำคัญจาก implementation (อ่านก่อนทำต่อ)

1. **Ω1 = θ/Θ** (ไม่ใช่ 1−θ/Θ ตามที่ OCR ของ Eq.2 อ่านได้)
   - ถอดจากตัวอย่าง A1 (θ=1,Θ=2→0.5) และ A2 (θ=2,Θ=2→1.0) และยืนยันซ้ำกับ EMP1 K-CPD (0.82)
   - θ = |L| − N (จำนวนเกรดที่ไม่ถูก assign), Θ = จำนวน gap ใน {γu, δi, γl} ที่ ≥ PVI ที่กว้างสุด
   - Ω1 = 1 ถ้า Θ = 0 ; clamp ให้อยู่ [0,1]

2. **σ ใน Ω3 ใช้ sample std (ddof=1)** ไม่ใช่ population — ตรวจแล้วให้ σ=5.07 (A1), 2.89 (A2)
   ตรงเปเปอร์ (โค้ดเดิม `run_cpd_theta_version.py` ใช้ np.std ddof=0 = ผิด)

3. **D_min/D_max ใน Ω2 นับเฉพาะ gap ระหว่างค่าที่อยู่ติดกัน** (ไม่รวม γu, γl); δi เป็น subset
   ของ gap เหล่านี้ → Ω2 ∈ [0,1] เสมอ

4. **⚠️ Ω′ DEGENERACY — finding ที่สำคัญที่สุด**: การหา argmax Ω′ อิสระข้าม k ทุกค่า
   **ยุบไปที่ N=2 แล้วได้ Ω′=1.0 แบบ trivial** เพราะเมื่อ N<3 → Ω2 ถูกบังคับ =1 และการแบ่ง
   2 กลุ่มที่ PVI เท่ากัน → σ=0 → Ω3=1 ด้วย
   → **Ω′ เป็น metric ไว้ "เปรียบเทียบวิธี" ที่ label budget |L| คงที่ ไม่ใช่ objective ที่จะ
   minimize อิสระข้าม k** เปเปอร์จึงเริ่มที่ |L| เกรดแล้ว "ลด" ตาม Requirement 1 เท่านั้น
   ไม่เคยค้นลงไปถึง N=2
   → เวลาใช้กับข้อมูลจริง ควร **fix |L|** (เช่น 8 เกรด) หรือจำกัด N≥3 / ทำแบบ reduce-from-|L|
   ตาม Algorithm 1 แทนการ sweep k เสรี

5. DP (SSE-optimal) ≠ Ω′-optimal — บน Table 3 exhaustive ชนะ DP ทุก k (เช่น k=2: 1.00 vs 0.22)
   เพราะ SSE กับ Ω′ ให้ partition คนละแบบ

---

## 4. คำถามที่ยังค้าง (Open decisions)
1. **Data**: ต้องได้ WGP.csv จริง (อัปโหลดใหม่เป็น attachment) — ตอนนี้ทดสอบด้วย Table 3
2. **k policy**: ยืนยันว่าจะ fix |L| (แนะนำ ตาม finding #4) หรือ sweep — ถ้า sweep ต้องกัน degeneracy
3. **U, L, grade symbols**: ปัจจุบัน default U=100, L=0, เกรด A,B+,B,C+,C,D+,D,F (8 ตัว) — ยืนยัน

## 5. สถานะ (Status)
- [x] อ่าน+สรุปเปเปอร์ทั้ง 2 + โค้ดผู้ใช้
- [x] Implement `calculate_omega_prime()` ตรง Eq.(2)
- [x] Verify กับ Table 3 (A1=0.08, A2=0.22) — PASS
- [x] Exhaustive + DP partitioner + benchmark
- [x] Verify exhaustive ≥ DP — PASS
- [ ] เข้าถึงข้อมูลจริง (WGP.csv) แล้วรันจริง
- [ ] ยืนยัน k policy (fix |L| vs sweep) กับผู้ใช้