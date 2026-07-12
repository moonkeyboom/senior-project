import './style.css'

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('cpdForm') as HTMLFormElement;
  const fileInput = document.getElementById('fileUpload') as HTMLInputElement;
  const fileDropArea = document.getElementById('fileDropArea');
  const fileMsg = document.querySelector('.file-msg') as HTMLElement;
  const submitBtn = document.getElementById('submitBtn') as HTMLButtonElement;
  const btnText = document.querySelector('.btn-text') as HTMLElement;
  const loader = document.querySelector('.loader') as HTMLElement;
  const errorMsg = document.getElementById('errorMsg') as HTMLElement;

  const inputSection = document.getElementById('inputSection') as HTMLElement;
  const resultSection = document.getElementById('resultSection') as HTMLElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;

  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // Tab switching logic
  tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = (e.currentTarget as HTMLElement).getAttribute('data-target');
      
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      (e.currentTarget as HTMLElement).classList.add('active');
      document.getElementById(target!)?.classList.add('active');
    });
  });

  // File Input handling for visual feedback
  fileInput.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      fileMsg.textContent = target.files[0].name;
    } else {
      fileMsg.textContent = 'คลิกเพื่อเลือกไฟล์ หรือ ลากและวางไฟล์ที่นี่';
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileDropArea?.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults (e: Event) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    fileDropArea?.addEventListener(eventName, () => fileDropArea.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    fileDropArea?.addEventListener(eventName, () => fileDropArea.classList.remove('dragover'), false);
  });

  fileDropArea?.addEventListener('drop', (e: DragEvent) => {
    const dt = e.dataTransfer;
    if (dt && dt.files.length) {
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event('change'));
    }
  });

  // Form Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    errorMsg.classList.add('hidden');
    
    if (!fileInput.files || fileInput.files.length === 0) {
      showError("กรุณาเลือกไฟล์ก่อนคำนวณ");
      return;
    }

    const labelsInput = document.getElementById('labelsInput') as HTMLInputElement;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('labels', labelsInput.value);

    // UI Loading state
    submitBtn.disabled = true;
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');

    try {
      // Connect to FastAPI backend
      const response = await fetch('http://localhost:8000/api/calculate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "เกิดข้อผิดพลาดในการประมวลผล");
      }

      const data = await response.json();
      renderResults(data);
      
      // Transition UI
      inputSection.classList.add('hidden');
      resultSection.classList.remove('hidden');

    } catch (err: any) {
      showError(err.message || "เกิดข้อผิดพลาดจากเซิร์ฟเวอร์");
    } finally {
      submitBtn.disabled = false;
      btnText.classList.remove('hidden');
      loader.classList.add('hidden');
    }
  });

  function showError(msg: string) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
  }

  function renderResults(data: any) {
    document.getElementById('omegaPrimeVal')!.textContent = data.omega_prime.toFixed(4);
    document.getElementById('omega1Val')!.textContent = data.omega1.toFixed(4);
    document.getElementById('omega2Val')!.textContent = data.omega2.toFixed(4);
    document.getElementById('omega3Val')!.textContent = data.omega3.toFixed(4);
    document.getElementById('totalRecords')!.textContent = data.n.toString();
    document.getElementById('methodBadge')!.textContent = data.method_name || "ไม่ทราบวิธีการ";

    const methodScoresList = document.getElementById('methodScoresList')!;
    methodScoresList.innerHTML = '';
    if (data.method_scores && data.method_scores.length > 0) {
      data.method_scores.forEach((m: any, index: number) => {
        const div = document.createElement('div');
        div.style.display = 'flex';
        div.style.justifyContent = 'space-between';
        div.style.padding = '10px 15px';
        div.style.background = index === 0 ? '#f0f4ff' : '#ffffff';
        div.style.border = index === 0 ? '1px solid #0f62fe' : '1px solid #eeeeee';
        div.style.borderRadius = '4px';
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = m.name;
        nameSpan.style.fontSize = '14px';
        if (index === 0) {
          nameSpan.style.color = '#0f62fe';
          nameSpan.style.fontWeight = '600';
        } else {
          nameSpan.style.color = '#333333';
        }
        
        const scoreSpan = document.createElement('span');
        scoreSpan.textContent = `Ω′ = ${m.omega_prime.toFixed(4)}`;
        scoreSpan.style.fontSize = '14px';
        if (index === 0) {
          scoreSpan.style.fontWeight = '600';
          scoreSpan.style.color = '#0f62fe';
        } else {
          scoreSpan.style.color = '#333333';
        }
        
        div.appendChild(nameSpan);
        div.appendChild(scoreSpan);
        methodScoresList.appendChild(div);
      });
    }

    const tbody = document.getElementById('clustersBody')!;
    tbody.innerHTML = ''; // clear old results

    data.clusters.forEach((cluster: any) => {
      const tr = document.createElement('tr');
      
      const tdGrade = document.createElement('td');
      tdGrade.textContent = cluster.grade;
      
      const tdMin = document.createElement('td');
      tdMin.textContent = cluster.min.toFixed(2);
      
      const tdMax = document.createElement('td');
      tdMax.textContent = cluster.max.toFixed(2);
      
      const tdAmount = document.createElement('td');
      tdAmount.textContent = cluster.amount.toString();
      
      tr.appendChild(tdGrade);
      tr.appendChild(tdMin);
      tr.appendChild(tdMax);
      tr.appendChild(tdAmount);
      tbody.appendChild(tr);
    });

    const allRecordsBody = document.getElementById('allRecordsBody')!;
    allRecordsBody.innerHTML = '';
    
    // For large datasets, building HTML string is faster than DOM manipulation per row
    let recordsHTML = '';
    data.labeled_records.forEach((record: any, idx: number) => {
      recordsHTML += `<tr>
        <td>${idx + 1}</td>
        <td>${record.score.toFixed(2)}</td>
        <td>${record.label}</td>
      </tr>`;
    });
    allRecordsBody.innerHTML = recordsHTML;
  }

  // Reset App State
  resetBtn.addEventListener('click', () => {
    resultSection.classList.add('hidden');
    inputSection.classList.remove('hidden');
    form.reset();
    fileMsg.textContent = 'คลิกเพื่อเลือกไฟล์ หรือ ลากและวางไฟล์ที่นี่';
  });
});
