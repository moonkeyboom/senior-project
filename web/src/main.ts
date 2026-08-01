import './style.css'

type LogLevel = 'info' | 'running' | 'success' | 'warning' | 'formula' | 'complete' | 'error'
type RunStatus = 'idle' | 'running' | 'complete' | 'error'

interface CalculationLogEvent {
  type: 'log'
  step: number
  level: LogLevel
  title: string
  detail: string
  elapsed_ms: number
  /** Method that produced this entry, for jumping from a results row. */
  trace?: string
  /** The one entry per method a results row should scroll to. */
  anchor?: boolean
}

interface CalculationErrorEvent {
  type: 'error'
  status_code: number
  message: string
  elapsed_ms: number
}

interface MethodScore {
  name: string
  trace: string
  rank: number
  omega_prime: number
  omega1: number
  omega2: number
  omega3: number
  sigma: number
  cluster_count: number
  cuts: number[]
  delta: number
  /** null when the best Ω′ is 0 — no meaningful ratio exists. */
  share: number | null
}

interface ClusterResult {
  grade: string
  min: number
  max: number
  amount: number
}

interface LabeledRecord {
  score: number
  label: string
}

interface RunNotice {
  level: 'warning' | 'info'
  text: string
}

interface ExhaustiveInfo {
  ran: boolean
  cancelled: boolean
  candidate_count: number
  candidate_formula: string
  mode: 'auto' | 'confirm' | 'refused'
}

interface CalculationResult {
  omega_prime: number
  omega1: number
  omega2: number
  omega3: number
  sigma: number
  clusters: ClusterResult[]
  labeled_records: LabeledRecord[]
  n: number
  method_name: string
  method_scores: MethodScore[]
  warnings: RunNotice[]
  source_filename: string
  score_column: string
  column_strategy: string
  labels: string
  label_budget: number
  cluster_count: number
  cuts: number[]
  removed_rows: number
  elapsed_ms: number
  exhaustive: ExhaustiveInfo
}

interface PreflightResponse {
  ok: true
  file_type: string
  source_filename: string
  rows: number
  n: number
  removed_rows: number
  score_column: string
  column_strategy: string
  score_min: number
  score_max: number
  labels: string[]
  label_budget: number
  exhaustive: {
    candidate_count: number
    candidate_formula: string
    estimated_seconds: number
    estimated_text: string
    mode: 'auto' | 'confirm' | 'refused'
    auto_limit: number
    confirm_limit: number
  }
  warnings: RunNotice[]
}

interface CalculationResultEvent {
  type: 'result'
  data: CalculationResult
}

type CalculationStreamEvent = CalculationLogEvent | CalculationErrorEvent | CalculationResultEvent

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('cpdForm') as HTMLFormElement
  const fileInput = document.getElementById('fileUpload') as HTMLInputElement
  const fileDropArea = document.getElementById('fileDropArea') as HTMLElement
  const fileMsg = document.querySelector('.file-msg') as HTMLElement
  const submitBtn = document.getElementById('submitBtn') as HTMLButtonElement
  const btnText = document.querySelector('.btn-text') as HTMLElement
  const loader = document.querySelector('.loader') as HTMLElement
  const errorMsg = document.getElementById('errorMsg') as HTMLElement
  const inputSection = document.getElementById('inputSection') as HTMLElement
  const resultSection = document.getElementById('resultSection') as HTMLElement
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement
  const labelsInput = document.getElementById('labelsInput') as HTMLInputElement
  const labelBudget = document.getElementById('labelBudget') as HTMLElement
  const labelsNote = document.getElementById('labelsNote') as HTMLElement
  const labelsError = document.getElementById('labelsError') as HTMLElement
  const errorText = document.getElementById('errorText') as HTMLElement
  const retryBtn = document.getElementById('retryBtn') as HTMLButtonElement
  const cancelBtn = document.getElementById('cancelBtn') as HTMLButtonElement
  const runSetup = document.getElementById('runSetup') as HTMLElement
  const fileAction = document.querySelector('.file-action') as HTMLElement

  const preflight = document.getElementById('preflight') as HTMLElement
  const preflightState = document.getElementById('preflightState') as HTMLElement
  const preflightFacts = document.getElementById('preflightFacts') as HTMLElement
  const preflightCost = document.getElementById('preflightCost') as HTMLElement
  const preflightConfirm = document.getElementById('preflightConfirm') as HTMLElement
  const forceExhaustive = document.getElementById('forceExhaustive') as HTMLInputElement
  const forceExhaustiveText = document.getElementById('forceExhaustiveText') as HTMLElement
  const preflightNotices = document.getElementById('preflightNotices') as HTMLUListElement

  const resultHeading = document.getElementById('resultHeading') as HTMLElement
  const runContext = document.getElementById('runContext') as HTMLElement
  const resultNotices = document.getElementById('resultNotices') as HTMLUListElement
  const methodsSubnote = document.getElementById('methodsSubnote') as HTMLElement
  const exportStatus = document.getElementById('exportStatus') as HTMLElement
  const exportXlsxBtn = document.getElementById('exportXlsxBtn') as HTMLButtonElement
  const exportCsvBtn = document.getElementById('exportCsvBtn') as HTMLButtonElement
  const copyBtn = document.getElementById('copyBtn') as HTMLButtonElement

  const logPanel = document.getElementById('executionLogPanel') as HTMLElement
  const logPanelToggle = document.getElementById('logPanelToggle') as HTMLButtonElement
  const logPanelClose = document.getElementById('logPanelClose') as HTMLButtonElement
  const logClearButton = document.getElementById('logClearButton') as HTMLButtonElement
  const logPanelBody = document.getElementById('logPanelBody') as HTMLElement
  const logTimeline = document.getElementById('logTimeline') as HTMLOListElement
  const logRunStatus = document.getElementById('logRunStatus') as HTMLElement
  const logStatusText = document.getElementById('logStatusText') as HTMLElement
  const logCount = document.getElementById('logCount') as HTMLElement
  const logElapsed = document.getElementById('logElapsed') as HTMLElement

  let logEntryCount = 0
  let latestElapsedMs = 0
  let calculationRunning = false
  let runController: AbortController | null = null
  let cancelledByUser = false
  let preflightData: PreflightResponse | null = null
  let preflightToken = 0
  let latestResult: CalculationResult | null = null

  const DEFAULT_FILE_ACTION = 'เลือกไฟล์คะแนน'
  const DEFAULT_FILE_MSG = 'หรือลากไฟล์มาวางที่นี่'
  const ACCEPTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']

  function syncSubmitAvailability() {
    submitBtn.disabled = calculationRunning || preflightData === null
  }

  syncSubmitAvailability()

  function iconMarkup(id: string) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('class', 'icon')
    svg.setAttribute('aria-hidden', 'true')
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use')
    use.setAttribute('href', `#${id}`)
    svg.appendChild(use)
    return svg
  }

  function renderNotices(target: HTMLUListElement, notices: RunNotice[]) {
    target.replaceChildren()
    target.classList.toggle('hidden', notices.length === 0)
    notices.forEach((notice) => {
      const item = document.createElement('li')
      item.className = 'notice'
      item.dataset.level = notice.level
      item.append(
        iconMarkup(notice.level === 'warning' ? 'i-alert' : 'i-info'),
      )
      const text = document.createElement('span')
      text.textContent = notice.text
      item.appendChild(text)
      target.appendChild(item)
    })
  }

  function renderFacts(target: HTMLElement, facts: [string, string][]) {
    target.replaceChildren()
    facts.forEach(([term, value]) => {
      const dt = document.createElement('dt')
      dt.textContent = term
      const dd = document.createElement('dd')
      dd.textContent = value
      target.append(dt, dd)
    })
  }

  // --- |L| validation -------------------------------------------------------
  // |L| is a fixed input, so the field has to be countable without ambiguity
  // before anything is sent. Duplicates and singletons are refused here and
  // again on the server.
  function readLabels() {
    const symbols = labelsInput.value
      .split(',')
      .map((symbol) => symbol.trim())
      .filter(Boolean)
    const seen = new Set<string>()
    const duplicates: string[] = []
    symbols.forEach((symbol) => {
      if (seen.has(symbol) && !duplicates.includes(symbol)) duplicates.push(symbol)
      seen.add(symbol)
    })

    let error = ''
    if (symbols.length === 0) {
      error = 'ยังไม่ได้ระบุระดับเกรด กรอกอย่างน้อย 2 ระดับ คั่นด้วยจุลภาค เช่น A, B, C'
    } else if (duplicates.length) {
      error = `ระดับเกรดซ้ำกัน: ${duplicates.join(', ')} — ลบตัวที่ซ้ำออก เพราะ |L| ต้องนับได้ชัดเจน`
    } else if (symbols.length < 2) {
      error = 'ต้องมีระดับเกรดอย่างน้อย 2 ระดับ ถ้าเหลือกลุ่มเดียว Ω′ จะเท่ากับ 1 เสมอ'
    }
    return { symbols, error }
  }

  function validateLabels() {
    const { symbols, error } = readLabels()
    labelBudget.textContent = String(symbols.length)
    labelsError.textContent = error
    labelsError.classList.toggle('hidden', !error)
    labelsInput.setAttribute('aria-invalid', String(Boolean(error)))
    labelsNote.textContent = symbols.length < 3
      ? 'N < 3 ทำให้ Ω2 = 1 โดยนิยาม — ค่า Ω′ จะสูงเกินจริง'
      : 'จำนวนเกรดคงที่ ระบบจะไม่ลดให้เอง'
    labelsNote.dataset.tone = symbols.length < 3 ? 'warning' : 'muted'
    return !error
  }

  // --- Preflight ------------------------------------------------------------
  // Quote n, the chosen column, and the exhaustive candidate count before the
  // user commits. C(n-1,k-1) grows fast enough that discovering the cost by
  // waiting is not an option.
  async function runPreflight() {
    const token = ++preflightToken
    const selectedFile = fileInput.files?.[0]
    const labelsValid = validateLabels()
    if (!selectedFile) {
      preflightData = null
      syncSubmitAvailability()
      preflight.classList.add('hidden')
      return
    }

    preflightData = null
    syncSubmitAvailability()
    preflight.classList.remove('hidden')
    preflight.dataset.state = 'loading'
    preflightState.textContent = 'กำลังอ่านไฟล์…'
    preflightCost.classList.add('hidden')
    preflightConfirm.classList.add('hidden')
    forceExhaustive.checked = false
    renderNotices(preflightNotices, [])
    renderFacts(preflightFacts, [])

    if (!labelsValid) {
      preflight.dataset.state = 'error'
      preflightState.textContent = 'แก้ระดับเกรดก่อน'
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('labels', labelsInput.value)

    try {
      const response = await fetch(`${API_BASE_URL}/api/preflight`, {
        method: 'POST',
        body: formData,
      })
      if (token !== preflightToken) return

      if (!response.ok) {
        const detail = await readErrorDetail(response)
        preflight.dataset.state = 'error'
        preflightState.textContent = 'ไฟล์นี้ยังคำนวณไม่ได้'
        renderNotices(preflightNotices, [{ level: 'warning', text: detail }])
        return
      }

      const data = await response.json() as PreflightResponse
      if (token !== preflightToken) return
      preflightData = data
      syncSubmitAvailability()
      preflight.dataset.state = 'ready'
      preflightState.textContent = 'พร้อมคำนวณ'

      renderFacts(preflightFacts, [
        ['ข้อมูล', `${data.n.toLocaleString()} คะแนน (จาก ${data.rows.toLocaleString()} แถว)`],
        ['คอลัมน์คะแนน', `${data.score_column} · ${data.column_strategy}`],
        ['ช่วงคะแนน', `${data.score_min} – ${data.score_max}`],
        ['label budget', `|L| = ${data.label_budget}`],
      ])

      const { candidate_count, candidate_formula, estimated_text, mode } = data.exhaustive
      preflightCost.classList.remove('hidden')
      preflightCost.dataset.mode = mode
      if (mode === 'refused') {
        preflightCost.textContent =
          `Exhaustive ต้องประเมิน ${candidate_count.toLocaleString()} partitions `
          + `(${candidate_formula}) ≈ ${estimated_text} — เกินเพดาน จะรันเฉพาะ DP `
          + 'ผลรอบนี้จึงไม่มี ground truth ยืนยัน'
      } else if (mode === 'confirm') {
        preflightCost.textContent =
          `Exhaustive ต้องประเมิน ${candidate_count.toLocaleString()} partitions `
          + `(${candidate_formula}) ≈ ${estimated_text}`
        preflightConfirm.classList.remove('hidden')
        forceExhaustiveText.textContent =
          `รัน Exhaustive ด้วย (ใช้เวลา ${estimated_text} · ยกเลิกกลางคันได้)`
      } else {
        preflightCost.textContent =
          `Exhaustive: ${candidate_count.toLocaleString()} partitions `
          + `(${candidate_formula}) ≈ ${estimated_text}`
      }

      renderNotices(preflightNotices, data.warnings)
    } catch (error) {
      if (token !== preflightToken) return
      preflight.dataset.state = 'error'
      preflightState.textContent = 'ตรวจสอบไฟล์ไม่สำเร็จ'
      renderNotices(preflightNotices, [{
        level: 'warning',
        text: error instanceof Error
          ? `เชื่อมต่อ CPD service ไม่ได้: ${error.message}`
          : 'เชื่อมต่อ CPD service ไม่ได้',
      }])
    }
  }

  async function readErrorDetail(response: Response) {
    try {
      const body = await response.json() as { detail?: string }
      if (body.detail) return body.detail
    } catch {
      // Fall through to the status-code message below.
    }
    return `เซิร์ฟเวอร์ตอบกลับด้วย HTTP ${response.status}`
  }

  function setLogPanelOpen(open: boolean) {
    document.body.classList.toggle('log-panel-open', open)
    logPanel.setAttribute('aria-hidden', String(!open))
    logPanelToggle.setAttribute('aria-expanded', String(open))
  }

  function setRunStatus(status: RunStatus, text: string) {
    logRunStatus.dataset.status = status
    logStatusText.textContent = text
  }

  function formatElapsed(milliseconds: number) {
    if (milliseconds >= 1000) {
      return `${(milliseconds / 1000).toFixed(2)} s`
    }
    return `${milliseconds.toFixed(1)} ms`
  }

  function appendLog(entry: Omit<CalculationLogEvent, 'type'>) {
    logEntryCount += 1
    latestElapsedMs = Math.max(latestElapsedMs, entry.elapsed_ms)
    logPanelBody.classList.add('has-logs')
    logCount.textContent = String(logEntryCount)
    logElapsed.textContent = formatElapsed(latestElapsedMs)

    const item = document.createElement('li')
    item.className = 'log-entry'
    item.dataset.level = entry.level
    item.id = `log-step-${entry.step}`
    // Anchors for the results table's "ดูสูตร" links.
    if (entry.trace) item.dataset.trace = entry.trace
    if (entry.anchor) item.dataset.anchor = 'true'

    const marker = document.createElement('span')
    marker.className = 'log-step-marker'
    marker.setAttribute('aria-hidden', 'true')
    if (entry.level === 'complete' || entry.level === 'success') {
      marker.textContent = '✓'
    } else if (entry.level === 'error') {
      marker.textContent = '!'
    } else {
      marker.textContent = String(entry.step).padStart(2, '0')
    }

    const content = document.createElement('div')
    content.className = 'log-entry-content'

    const heading = document.createElement('div')
    heading.className = 'log-entry-heading'

    const title = document.createElement('span')
    title.className = 'log-entry-title'
    title.textContent = entry.title

    const time = document.createElement('span')
    time.className = 'log-entry-time'
    time.textContent = `+${formatElapsed(entry.elapsed_ms)}`

    const detail = document.createElement('p')
    detail.className = 'log-entry-detail'
    detail.textContent = entry.detail

    heading.append(title, time)
    content.append(heading, detail)
    item.append(marker, content)
    logTimeline.appendChild(item)

    requestAnimationFrame(() => {
      logPanelBody.scrollTo({ top: logPanelBody.scrollHeight, behavior: 'smooth' })
    })
  }

  function appendErrorLog(message: string, elapsedMs: number) {
    appendLog({
      step: logEntryCount + 1,
      level: 'error',
      title: 'การประมวลผลล้มเหลว',
      detail: message,
      elapsed_ms: elapsedMs,
    })
  }

  function clearExecutionLog(resetStatus = true) {
    logTimeline.replaceChildren()
    logEntryCount = 0
    latestElapsedMs = 0
    logCount.textContent = '0'
    logElapsed.textContent = '0.0 ms'
    logPanelBody.classList.remove('has-logs')
    if (resetStatus) {
      setRunStatus('idle', 'พร้อมเริ่มทำงาน')
    }
  }

  logPanelToggle.addEventListener('click', () => setLogPanelOpen(true))
  logPanelClose.addEventListener('click', () => setLogPanelOpen(false))
  logClearButton.addEventListener('click', () => clearExecutionLog(!calculationRunning))
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.classList.contains('log-panel-open')) {
      setLogPanelOpen(false)
      logPanelToggle.focus({ preventScroll: true })
    }
  })

  const tabButtons = document.querySelectorAll<HTMLButtonElement>('.tab-btn')
  const tabContents = document.querySelectorAll<HTMLElement>('.tab-content')
  tabButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.dataset.target
      tabButtons.forEach((tabButton) => tabButton.classList.remove('active'))
      tabContents.forEach((content) => content.classList.remove('active'))
      button.classList.add('active')
      if (target) {
        document.getElementById(target)?.classList.add('active')
      }
    })
  })

  fileInput.addEventListener('change', () => {
    const selectedFile = fileInput.files?.[0]
    fileDropArea.dataset.state = selectedFile ? 'selected' : 'empty'
    fileAction.textContent = selectedFile?.name || DEFAULT_FILE_ACTION
    fileMsg.textContent = selectedFile
      ? 'ไฟล์พร้อมตรวจสอบ · คลิกเพื่อเลือกไฟล์อื่น'
      : DEFAULT_FILE_MSG
    runSetup.classList.toggle('hidden', !selectedFile)
    hideError()
    void runPreflight()
  })

  let labelsDebounce = 0
  labelsInput.addEventListener('input', () => {
    preflightToken += 1
    preflightData = null
    syncSubmitAvailability()
    validateLabels()
    window.clearTimeout(labelsDebounce)
    labelsDebounce = window.setTimeout(() => void runPreflight(), 400)
  })

  ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
    fileDropArea.addEventListener(eventName, preventDefaults, false)
  })

  function preventDefaults(event: Event) {
    event.preventDefault()
    event.stopPropagation()
  }

  ;['dragenter', 'dragover'].forEach((eventName) => {
    fileDropArea.addEventListener(eventName, () => fileDropArea.classList.add('dragover'))
  })

  ;['dragleave', 'drop'].forEach((eventName) => {
    fileDropArea.addEventListener(eventName, () => fileDropArea.classList.remove('dragover'))
  })

  fileDropArea.addEventListener('drop', (event: DragEvent) => {
    const droppedFiles = event.dataTransfer?.files
    const dropped = droppedFiles?.[0]
    if (!dropped) return

    // The `accept` attribute only filters the picker dialog; a drop bypasses it
    // entirely, so the same rule has to be applied by hand here.
    const name = dropped.name.toLowerCase()
    if (!ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
      showError(
        `ไฟล์ “${dropped.name}” ไม่ใช่ .csv หรือ .xlsx — ระบบรองรับเฉพาะไฟล์ตารางสองรูปแบบนี้`,
        { retry: false },
      )
      return
    }

    fileInput.files = droppedFiles
    fileInput.dispatchEvent(new Event('change'))
  })

  async function readCalculationStream(response: Response): Promise<CalculationResult> {
    if (!response.body) {
      throw new Error('เบราว์เซอร์ไม่รองรับ response streaming')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalResult: CalculationResult | null = null

    const processLine = (line: string) => {
      if (!line.trim()) return

      let event: CalculationStreamEvent
      try {
        event = JSON.parse(line) as CalculationStreamEvent
      } catch {
        throw new Error(`ไม่สามารถอ่าน execution event ได้: ${line.slice(0, 120)}`)
      }

      if (event.type === 'log') {
        appendLog(event)
      } else if (event.type === 'result') {
        finalResult = event.data
      } else if (event.type === 'error') {
        appendErrorLog(event.message, event.elapsed_ms)
        throw new Error(event.message)
      }
    }

    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      lines.forEach(processLine)
      if (done) break
    }

    processLine(buffer)
    if (!finalResult) {
      throw new Error('การเชื่อมต่อสิ้นสุดก่อนที่จะได้รับผลการคำนวณ')
    }
    return finalResult
  }

  function setRunningUI(running: boolean) {
    calculationRunning = running
    syncSubmitAvailability()
    submitBtn.dataset.state = running ? 'running' : 'idle'
    submitBtn.setAttribute('aria-busy', String(running))
    btnText.classList.toggle('hidden', running)
    loader.classList.toggle('hidden', !running)
    cancelBtn.classList.toggle('hidden', !running)
    fileInput.disabled = running
    labelsInput.disabled = running
  }

  cancelBtn.addEventListener('click', () => {
    if (!runController) return
    cancelledByUser = true
    cancelBtn.disabled = true
    runController.abort()
    appendLog({
      step: logEntryCount + 1,
      level: 'warning',
      title: 'ผู้ใช้ยกเลิกการคำนวณ',
      detail: 'ปิดการเชื่อมต่อ stream — เซิร์ฟเวอร์จะหยุดค้นหาภายในไม่กี่มิลลิวินาที',
      elapsed_ms: latestElapsedMs,
    })
  })

  retryBtn.addEventListener('click', () => {
    hideError()
    form.requestSubmit()
  })

  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    hideError()

    const selectedFile = fileInput.files?.[0]
    if (!selectedFile) {
      showError('กรุณาเลือกไฟล์ก่อนคำนวณ', { retry: false })
      clearExecutionLog()
      setLogPanelOpen(true)
      appendErrorLog('ไม่ได้เลือกไฟล์ CSV หรือ Excel สำหรับการคำนวณ', 0)
      setRunStatus('error', 'ข้อมูลไม่ครบถ้วน')
      return
    }

    if (!validateLabels()) {
      showError(labelsError.textContent || 'ระดับเกรดไม่ถูกต้อง', { retry: false })
      labelsInput.focus()
      return
    }

    if (!preflightData) {
      showError('รอให้ระบบตรวจสอบไฟล์และต้นทุนการคำนวณให้สำเร็จก่อน', { retry: false })
      void runPreflight()
      return
    }

    const wantsExhaustive = preflightData.exhaustive.mode !== 'confirm' || forceExhaustive.checked
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('labels', labelsInput.value)
    formData.append('force_exhaustive', String(wantsExhaustive))

    clearExecutionLog(false)
    setLogPanelOpen(true)
    setRunStatus('running', 'กำลังประมวลผล…')
    cancelledByUser = false
    cancelBtn.disabled = false
    runController = new AbortController()
    setRunningUI(true)
    appendLog({
      step: 0,
      level: 'running',
      title: 'เตรียมส่งข้อมูลไปยัง CPD service',
      detail: `POST /api/calculate/stream · ${selectedFile.name}`,
      elapsed_ms: 0,
    })

    try {
      const response = await fetch(`${API_BASE_URL}/api/calculate/stream`, {
        method: 'POST',
        body: formData,
        signal: runController.signal,
      })

      if (!response.ok) {
        throw new Error(await readErrorDetail(response))
      }

      const data = await readCalculationStream(response)
      latestResult = data
      renderResults(data)
      inputSection.classList.add('hidden')
      resultSection.classList.remove('hidden')
      setRunStatus('complete', 'ประมวลผลสำเร็จ')
      window.scrollTo({ top: 0, behavior: 'smooth' })
      resultHeading.focus({ preventScroll: true })
    } catch (error: unknown) {
      if (cancelledByUser) {
        setRunStatus('idle', 'ยกเลิกแล้ว')
        showError(
          'ยกเลิกการคำนวณแล้ว ข้อมูลและระดับเกรดที่กรอกไว้ยังอยู่ กดคำนวณอีกครั้งได้เลย',
          { retry: true, tone: 'info' },
        )
      } else {
        const message = error instanceof Error ? error.message : 'เกิดข้อผิดพลาดจากเซิร์ฟเวอร์'
        const alreadyLogged =
          logTimeline.lastElementChild?.getAttribute('data-level') === 'error'
        if (!alreadyLogged) {
          appendErrorLog(message, latestElapsedMs)
        }
        setRunStatus('error', 'ประมวลผลไม่สำเร็จ')
        showError(message, { retry: true })
      }
    } finally {
      runController = null
      setRunningUI(false)
    }
  })

  function showError(
    message: string,
    options: { retry?: boolean; tone?: 'error' | 'info' } = {},
  ) {
    errorText.textContent = message
    errorMsg.dataset.tone = options.tone ?? 'error'
    retryBtn.classList.toggle('hidden', options.retry === false)
    errorMsg.classList.remove('hidden')
  }

  function hideError() {
    errorMsg.classList.add('hidden')
  }

  // --- The comparison ---------------------------------------------------------
  // Ω′ alone says nothing; the ranking against the baselines is the product.
  // Every row carries rank, the score, the gap to the best, a bar, and a link
  // to the derivation that produced it.
  function jumpToTrace(trace: string) {
    setLogPanelOpen(true)
    const target = logTimeline.querySelector<HTMLElement>(
      `[data-trace="${CSS.escape(trace)}"][data-anchor="true"]`,
    ) ?? logTimeline.querySelector<HTMLElement>(`[data-trace="${CSS.escape(trace)}"]`)
    if (!target) return
    requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'center', behavior: 'smooth' })
      target.classList.remove('log-entry-flash')
      void target.offsetWidth
      target.classList.add('log-entry-flash')
    })
  }

  function cell(className: string, text: string) {
    const td = document.createElement('td')
    td.className = className
    td.textContent = text
    return td
  }

  function renderMethodTable(data: CalculationResult) {
    const tbody = document.getElementById('methodScoresList')!
    const note = document.getElementById('methodTableNote')!
    tbody.replaceChildren()

    const methods = data.method_scores ?? []
    const best = methods[0]
    // Ω′ = 0 for every method is a real outcome (a DP-only run on tc1), and
    // ratios against zero are undefined — drop the bars rather than draw a row
    // of empty tracks that read as "no data".
    const ratiosDefined = Boolean(best) && best.omega_prime > 0

    methods.forEach((method) => {
      const row = document.createElement('tr')
      // A single result is not a ranking — don't dress it as a winner.
      const isWinner = methods.length > 1 && method.rank === 1
      if (isWinner) row.dataset.winner = 'true'

      row.append(
        cell('method-rank', String(method.rank)),
        cell('method-name', method.name),
        cell('method-omega', method.omega_prime.toFixed(4)),
      )

      // -0.0000 is arithmetic noise, not a gap.
      const delta = Object.is(method.delta, -0) ? 0 : method.delta
      row.appendChild(cell(
        'method-delta',
        method.rank === 1 ? '—' : `${delta > 0 ? '+' : ''}${delta.toFixed(4)}`,
      ))

      const barCell = document.createElement('td')
      barCell.className = 'method-bar-cell'
      if (ratiosDefined && method.share !== null) {
        const track = document.createElement('div')
        track.className = 'method-bar'
        const fill = document.createElement('span')
        fill.style.width = `${Math.max(0, Math.min(1, method.share)) * 100}%`
        track.appendChild(fill)
        barCell.appendChild(track)
        barCell.title = `${(method.share * 100).toFixed(1)}% ของค่าที่ดีที่สุด`
      }
      row.appendChild(barCell)

      const traceCell = document.createElement('td')
      traceCell.className = 'method-trace'
      if (method.trace) {
        const link = document.createElement('button')
        link.type = 'button'
        link.className = 'trace-link'
        link.textContent = 'ดูสูตร →'
        link.setAttribute('aria-label', `ดูการคำนวณ Ω′ ของ ${method.name}`)
        link.addEventListener('click', () => jumpToTrace(method.trace))
        traceCell.appendChild(link)
      }
      row.appendChild(traceCell)

      tbody.appendChild(row)
    })

    if (methods.length === 1) {
      note.textContent =
        'มีวิธีเดียวในรอบนี้ จึงยังไม่ใช่การเปรียบเทียบ — '
        + 'รัน Exhaustive เพื่อให้มี ground truth มาเทียบ'
      note.classList.remove('hidden')
    } else if (!ratiosDefined) {
      note.textContent =
        'ทุกวิธีได้ Ω′ = 0 จึงไม่มีสัดส่วนให้เทียบ ดูองค์ประกอบด้านล่างว่าตัวไหนเป็นศูนย์'
      note.classList.remove('hidden')
    } else {
      note.classList.add('hidden')
    }
  }

  function renderResults(data: CalculationResult) {
    exportStatus.classList.add('hidden')
    document.getElementById('omegaPrimeVal')!.textContent = data.omega_prime.toFixed(4)
    // Ω′ = Ω1 × Ω2 × Ω3, so one zero factor annihilates the product. Mark which.
    ;([['omega1Val', 'factor1', data.omega1],
       ['omega2Val', 'factor2', data.omega2],
       ['omega3Val', 'factor3', data.omega3]] as const).forEach(
      ([valueId, factorId, value]) => {
        document.getElementById(valueId)!.textContent = value.toFixed(4)
        document.getElementById(factorId)!.dataset.zero = String(value === 0)
      },
    )
    document.getElementById('totalRecords')!.textContent = data.n.toString()
    document.getElementById('methodBadge')!.textContent = data.method_name || 'ไม่ทราบวิธีการ'

    renderFacts(runContext, [
      ['ไฟล์', data.source_filename],
      ['คอลัมน์คะแนน', `${data.score_column} · ${data.column_strategy}`],
      ['ระดับเกรด', `${data.labels} (|L| = ${data.label_budget})`],
      ['ผลลัพธ์', `N = ${data.cluster_count} กลุ่ม · cuts = [${data.cuts.join(', ')}]`],
      ['เวลาที่ใช้', formatElapsed(data.elapsed_ms)],
    ])
    renderNotices(resultNotices, data.warnings ?? [])

    methodsSubnote.textContent = data.exhaustive.ran
      ? `เทียบ ${data.method_scores.length} วิธี · Exhaustive ตรวจครบ `
        + `${data.exhaustive.candidate_count.toLocaleString()} partitions`
      : `เทียบ ${data.method_scores.length} วิธี · ไม่ได้รัน Exhaustive `
        + '(ยังไม่มี ground truth ยืนยันผลนี้)'

    renderMethodTable(data)

    const clustersBody = document.getElementById('clustersBody')!
    clustersBody.replaceChildren()
    data.clusters.forEach((cluster) => {
      const row = document.createElement('tr')
      ;[cluster.grade, cluster.min.toFixed(2), cluster.max.toFixed(2), String(cluster.amount)]
        .forEach((value) => {
          const cell = document.createElement('td')
          cell.textContent = value
          row.appendChild(cell)
        })
      clustersBody.appendChild(row)
    })

    const allRecordsBody = document.getElementById('allRecordsBody')!
    const recordsFragment = document.createDocumentFragment()
    data.labeled_records.forEach((record, index) => {
      const row = document.createElement('tr')
      ;[String(index + 1), record.score.toFixed(2), record.label].forEach((value) => {
        const cell = document.createElement('td')
        cell.textContent = value
        row.appendChild(cell)
      })
      recordsFragment.appendChild(row)
    })
    allRecordsBody.replaceChildren(recordsFragment)
  }

  // --- Export ---------------------------------------------------------------
  // The job ends at a file, not at the screen. Both formats are built from one
  // server-side builder so the CSV and the XLSX can never carry a different
  // manifest for the same run.
  function buildExportPayload(format: 'csv' | 'xlsx', data: CalculationResult) {
    return {
      format,
      manifest: {
        source_filename: data.source_filename,
        score_column: data.score_column,
        column_strategy: data.column_strategy,
        labels: data.labels,
        label_budget: data.label_budget,
        n: data.n,
        removed_rows: data.removed_rows,
        cluster_count: data.cluster_count,
        method_name: data.method_name,
        omega_prime: data.omega_prime,
        omega1: data.omega1,
        omega2: data.omega2,
        omega3: data.omega3,
        sigma: data.sigma,
        cuts: data.cuts,
        elapsed_ms: data.elapsed_ms,
        exhaustive_ran: data.exhaustive.ran,
        exhaustive_candidates: data.exhaustive.candidate_count,
        method_scores: data.method_scores,
        warnings: (data.warnings ?? []).map((notice) => notice.text),
      },
      clusters: data.clusters,
      records: data.labeled_records,
    }
  }

  function setExportStatus(message: string, tone: 'info' | 'error' | 'success') {
    exportStatus.textContent = message
    exportStatus.dataset.tone = tone
    exportStatus.classList.remove('hidden')
  }

  async function downloadExport(format: 'csv' | 'xlsx', trigger: HTMLButtonElement) {
    if (!latestResult) return
    trigger.disabled = true
    setExportStatus('กำลังสร้างไฟล์…', 'info')
    try {
      const response = await fetch(`${API_BASE_URL}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildExportPayload(format, latestResult)),
      })
      if (!response.ok) throw new Error(await readErrorDetail(response))

      const blob = await response.blob()
      const disposition = response.headers.get('Content-Disposition') || ''
      const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition)?.[1]
      const plain = /filename="([^"]+)"/i.exec(disposition)?.[1]
      const filename = encoded
        ? decodeURIComponent(encoded)
        : plain || `cpd-result.${format}`

      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
      setExportStatus(`บันทึกไฟล์ ${filename} แล้ว`, 'success')
    } catch (error) {
      setExportStatus(
        error instanceof Error
          ? `ส่งออกไม่สำเร็จ: ${error.message} — ลองใหม่ หรือใช้ปุ่มคัดลอกตารางแทน`
          : 'ส่งออกไม่สำเร็จ ลองใหม่ หรือใช้ปุ่มคัดลอกตารางแทน',
        'error',
      )
    } finally {
      trigger.disabled = false
    }
  }

  exportXlsxBtn.addEventListener('click', () => void downloadExport('xlsx', exportXlsxBtn))
  exportCsvBtn.addEventListener('click', () => void downloadExport('csv', exportCsvBtn))

  copyBtn.addEventListener('click', async () => {
    if (!latestResult) return
    const header = 'rank\tscore\tgrade'
    const rows = latestResult.labeled_records.map(
      (record, index) => `${index + 1}\t${record.score}\t${record.label}`,
    )
    const text = [header, ...rows].join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setExportStatus(
        `คัดลอก ${rows.length.toLocaleString()} แถวแล้ว วางลงสเปรดชีตได้ทันที`,
        'success',
      )
    } catch {
      setExportStatus(
        'เบราว์เซอร์ไม่อนุญาตให้เข้าถึงคลิปบอร์ด ใช้ปุ่มดาวน์โหลดแทน',
        'error',
      )
    }
  })

  resetBtn.addEventListener('click', () => {
    resultSection.classList.add('hidden')
    inputSection.classList.remove('hidden')
    // Deliberately keep the file and the label list: the workflow is the same
    // dataset re-run at a different |L|, and form.reset() would destroy both.
    hideError()
    exportStatus.classList.add('hidden')
    latestResult = null
    clearExecutionLog()
    setLogPanelOpen(false)
    void runPreflight()
    fileInput.focus({ preventScroll: true })
  })

  validateLabels()
})
