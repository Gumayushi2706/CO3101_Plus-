/**
 * CO3101 RAG STUDIO - Client Application Logic
 * Ultra-fast, zero-overhead Vanilla JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global Elements
  const tabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const toastContainer = document.getElementById('toastContainer');
  const navDocCount = document.getElementById('navDocCount');

  // Chat elements
  const chatViewport = document.getElementById('chatViewport');
  const welcomeCard = document.getElementById('welcomeCard');
  const messagesList = document.getElementById('messagesList');
  const chatForm = document.getElementById('chatForm');
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const chatPartitionSelect = document.getElementById('chatPartitionSelect');
  const clearChatBtn = document.getElementById('clearChatBtn');

  // Multi-turn session ID
  let currentSessionId = 'session_' + Math.random().toString(36).substring(2, 10);

  // Docs & Partition elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const documentsTableBody = document.getElementById('documentsTableBody');
  const refreshDocsBtn = document.getElementById('refreshDocsBtn');
  const rebuildIndexBtn = document.getElementById('rebuildIndexBtn');
  const docPartitionSelect = document.getElementById('docPartitionSelect');
  const createPartitionBtn = document.getElementById('createPartitionBtn');
  const deletePartitionBtn = document.getElementById('deletePartitionBtn');
  const activeUploadPartitionName = document.getElementById('activeUploadPartitionName');
  const docTableTitle = document.getElementById('docTableTitle');

  // Config elements
  const cfgChunkSize = document.getElementById('cfgChunkSize');
  const cfgChunkOverlap = document.getElementById('cfgChunkOverlap');
  const saveConfigBtn = document.getElementById('saveConfigBtn');

  const valChunkSize = document.getElementById('valChunkSize');
  const valChunkOverlap = document.getElementById('valChunkOverlap');

  const clearInputBtn = document.getElementById('clearInputBtn');

  // Exact trash icon matching uploaded design
  const TRASH_ICON_SVG = `
    <svg class="trash-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9 6.8V4.8a1.5 1.5 0 0 1 1.5-1.5h3a1.5 1.5 0 0 1 1.5 1.5v2" />
      <line x1="4" y1="6.8" x2="20" y2="6.8" />
      <path d="M5.5 6.8l1.2 13a2 2 0 0 0 2 1.8h6.6a2 2 0 0 0 2-1.8l1.2-13" />
      <line x1="9.5" y1="10.5" x2="9.5" y2="17.5" />
      <line x1="12" y1="10.5" x2="12" y2="17.5" />
      <line x1="14.5" y1="10.5" x2="14.5" y2="17.5" />
    </svg>
  `;

  // State
  let isGenerating = false;

  // -------------------------------------------------------------
  // 1. Toast Notification System
  // -------------------------------------------------------------
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';

    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(30px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // -------------------------------------------------------------
  // 2. Navigation Tabs
  // -------------------------------------------------------------
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'docsTab') {
        loadPartitions();
        loadDocuments();
        fetchStats();
      } else if (targetId === 'chatTab') {
        loadPartitions();
      }
    });
  });

  // -------------------------------------------------------------
  // 3. Stats & Initial Fetch
  // -------------------------------------------------------------
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) throw new Error('Không thể tải thông tin hệ thống');
      const data = await res.json();

      if (navDocCount) navDocCount.textContent = data.raw_files_count;

      // Update config inputs
      if (cfgChunkSize) {
        cfgChunkSize.value = data.chunk_size;
        valChunkSize.textContent = `${data.chunk_size} ký tự`;
      }
      if (cfgChunkOverlap) {
        cfgChunkOverlap.value = data.chunk_overlap;
        valChunkOverlap.textContent = `${data.chunk_overlap} ký tự`;
      }
    } catch (err) {
      console.error('Lỗi khi tải thông tin hệ thống:', err);
    }
  }

  // -------------------------------------------------------------
  // 4. Chat Workflow & Message Rendering
  // -------------------------------------------------------------
  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function appendUserMessage(text, turnId) {
    if (welcomeCard) welcomeCard.style.display = 'none';

    const row = document.createElement('div');
    row.className = 'msg-row user-msg';
    if (turnId) row.setAttribute('data-turn-id', turnId);
    row.innerHTML = `
      <div class="msg-avatar">Bạn</div>
      <div class="msg-bubble">
        <div class="msg-text">${escapeHtml(text)}</div>
        <button type="button" class="btn-delete-question" data-turn-id="${turnId}" title="Xóa câu hỏi này">
          ${TRASH_ICON_SVG}
        </button>
      </div>
    `;
    messagesList.appendChild(row);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'msg-row assistant-msg typing-row';
    row.id = 'typingIndicator';
    row.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    messagesList.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeTypingIndicator() {
    const ind = document.getElementById('typingIndicator');
    if (ind) ind.remove();
  }

  function appendAssistantMessage(data, turnId) {
    removeTypingIndicator();

    const row = document.createElement('div');
    row.className = 'msg-row assistant-msg';
    if (turnId) row.setAttribute('data-turn-id', turnId);

    // Parse markdown if marked is loaded
    let parsedAnswer = data.answer;
    if (window.marked && typeof window.marked.parse === 'function') {
      parsedAnswer = window.marked.parse(data.answer);
    } else {
      parsedAnswer = `<p>${escapeHtml(data.answer)}</p>`;
    }

    // Render static citation badges [i] (không gắn hành vi nhảy/click)
    parsedAnswer = parsedAnswer.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, '<span class="citation-badge">[$1]</span>');

    const telemetryHtml = `
      <div class="telemetry-row">
        <div class="tel-pill latency">Thời gian phản hồi: <strong>${data.latency_seconds}s</strong></div>
        <div class="tel-pill">Đạt ngưỡng: <strong>${data.filtered_count} chunks</strong></div>
      </div>
    `;

    // Cửa sổ nguồn thông tin liên quan (Evidence chunks)
    let evidenceHtml = '';
    const evidences = data.evidences || [];
    if (evidences.length > 0) {
      const cardsList = evidences.map((ev, idx) => {
        const citationId = ev.citation_id || (idx + 1);
        const pageText = ev.page ? `Trang ${ev.page}` : 'N/A';
        const scorePercent = Math.min(100, Math.max(0, Math.round(ev.score * 100)));
        const partTag = ev.partition ? `[${escapeHtml(ev.partition)}] ` : '';

        return `
          <div class="evidence-card" id="ev-card-${citationId}">
            <div class="ev-meta-line">
              <span class="ev-badge-source">#${citationId} ${partTag}Nguồn: ${escapeHtml(ev.source)} (${pageText})</span>
              <div class="ev-score-container" title="Cosine Relevance Score τ">
                <div class="score-bar"><div class="score-fill" style="width: ${scorePercent}%"></div></div>
                <span class="score-val">τ: ${ev.score.toFixed(4)}</span>
              </div>
            </div>
            <div class="ev-content">${escapeHtml(ev.chunk)}</div>
          </div>
        `;
      }).join('');

      evidenceHtml = `
        <div class="evidence-section">
          <div class="evidence-header" onclick="this.parentElement.classList.toggle('open')">
            <span>Nguồn thông tin liên quan (${evidences.length} chunks)</span>
            <span class="ev-toggle-icon">▼</span>
          </div>
          <div class="evidence-cards-list">
            ${cardsList}
          </div>
        </div>
      `;
    } else {
      evidenceHtml = `
        <div class="evidence-section">
          <div class="evidence-header" onclick="this.parentElement.classList.toggle('open')">
            <span>Nguồn thông tin liên quan (0 chunks)</span>
            <span class="ev-toggle-icon">▼</span>
          </div>
          <div class="evidence-cards-list">
            <div class="evidence-card" style="border-left-color: #64748b;">
              <div class="ev-content" style="color: #94a3b8;">Không có đoạn trích nào đạt ngưỡng tương quan τ. Hệ thống kích hoạt cơ chế Anti-hallucination.</div>
            </div>
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble">
        <div class="answer-content">${parsedAnswer}</div>
        ${telemetryHtml}
        ${evidenceHtml}
      </div>
    `;

    messagesList.appendChild(row);
    scrollToBottom();
  }

  // Handle Query Submission
  async function submitQuery(questionText) {
    const text = questionText.trim();
    if (!text || isGenerating) return;

    const turnId = 'turn_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);

    isGenerating = true;
    sendBtn.disabled = true;
    appendUserMessage(text, turnId);
    queryInput.value = '';
    queryInput.style.height = 'auto';
    if (clearInputBtn) clearInputBtn.style.display = 'none';

    appendTypingIndicator();

    try {
      const selectedModel = 'gemini-3.5-flash-lite';
      const selectedPartition = chatPartitionSelect ? chatPartitionSelect.value : 'all';
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          model_name: selectedModel,
          session_id: currentSessionId,
          partition: selectedPartition,
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Lỗi xử lý truy vấn');
      }

      const data = await res.json();
      appendAssistantMessage(data, turnId);
    } catch (err) {
      console.error(err);
      removeTypingIndicator();
      showToast(err.message, 'error');

      const errRow = document.createElement('div');
      errRow.className = 'msg-row assistant-msg';
      errRow.setAttribute('data-turn-id', turnId);
      errRow.innerHTML = `
        <div class="msg-avatar" style="background: #ef4444;">!</div>
        <div class="msg-bubble" style="border-color: rgba(239, 68, 68, 0.4); color: #fca5a5;">
          <strong>Đã xảy ra lỗi:</strong> ${escapeHtml(err.message)}
        </div>
      `;
      messagesList.appendChild(errRow);
      scrollToBottom();
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
    }
  }

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitQuery(queryInput.value);
  });

  // Auto-resize, Enter key, and clear input behavior
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery(queryInput.value);
    }
  });

  queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
    if (clearInputBtn) {
      clearInputBtn.style.display = queryInput.value.trim().length > 0 ? 'flex' : 'none';
    }
  });

  if (clearInputBtn) {
    clearInputBtn.addEventListener('click', () => {
      queryInput.value = '';
      queryInput.style.height = 'auto';
      clearInputBtn.style.display = 'none';
      queryInput.focus();
    });
  }

  // Clear All Chat
  clearChatBtn.addEventListener('click', () => {
    messagesList.innerHTML = '';
    if (welcomeCard) welcomeCard.style.display = 'block';
    if (currentSessionId) {
      fetch('/api/chat/history/' + currentSessionId, { method: 'DELETE' }).catch(() => {});
    }
    currentSessionId = 'session_' + Math.random().toString(36).substring(2, 10);
    showToast('Đã xóa toàn bộ cuộc hội thoại', 'info');
  });

  // Delete Individual Question & Answer
  messagesList.addEventListener('click', (e) => {
    const deleteBtn = e.target.closest('.btn-delete-question');
    if (!deleteBtn) return;
    const turnId = deleteBtn.getAttribute('data-turn-id');
    if (!turnId) return;

    const matched = messagesList.querySelectorAll(`[data-turn-id="${turnId}"]`);
    matched.forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(-10px)';
      el.style.transition = 'all 0.25s ease';
      setTimeout(() => {
        el.remove();
        if (!messagesList.querySelector('.msg-row')) {
          if (welcomeCard) welcomeCard.style.display = 'block';
        }
      }, 250);
    });
    showToast('Đã xóa câu hỏi khỏi cuộc trò chuyện', 'info');
  });

  // -------------------------------------------------------------
  // 5. Partition & Document Management
  // -------------------------------------------------------------
  let cachedPartitions = [];

  async function loadPartitions() {
    try {
      const res = await fetch('/api/partitions');
      if (!res.ok) throw new Error('Không thể tải danh sách phân vùng');
      const data = await res.json();
      cachedPartitions = data.partitions || [];

      // Save currently selected values if possible
      const curChatVal = chatPartitionSelect ? chatPartitionSelect.value : 'all';
      const curDocVal = docPartitionSelect ? docPartitionSelect.value : 'all';

      // 1. Update chatPartitionSelect
      if (chatPartitionSelect) {
        chatPartitionSelect.innerHTML = cachedPartitions.map(p => {
          const name = p.is_default ? 'Default' : p.name;
          return `<option value="${escapeHtml(p.id)}">${escapeHtml(name)}</option>`;
        }).join('');
        if (cachedPartitions.some(p => p.id === curChatVal)) {
          chatPartitionSelect.value = curChatVal;
        } else {
          chatPartitionSelect.value = 'all';
        }
      }

      // 2. Update docPartitionSelect
      if (docPartitionSelect) {
        docPartitionSelect.innerHTML = cachedPartitions.map(p => {
          const name = p.is_default ? 'Default' : p.name;
          return `<option value="${escapeHtml(p.id)}">${escapeHtml(name)}</option>`;
        }).join('');
        if (cachedPartitions.some(p => p.id === curDocVal)) {
          docPartitionSelect.value = curDocVal;
        } else {
          docPartitionSelect.value = 'all';
        }
      }

      updatePartitionUI();
    } catch (err) {
      console.error(err);
    }
  }

  function updatePartitionUI() {
    if (!docPartitionSelect) return;
    const selectedId = docPartitionSelect.value;
    const currentPart = cachedPartitions.find(p => p.id === selectedId) || { file_count: 0, vector_count: 0, is_default: false, is_all: false };

    // Toggle delete button: Only user-defined partitions can be deleted (not 'all' and not 'default')
    if (deletePartitionBtn) {
      if (selectedId !== 'all' && selectedId !== 'default') {
        deletePartitionBtn.style.display = 'inline-flex';
      } else {
        deletePartitionBtn.style.display = 'none';
      }
    }

    // Update active dropzone label
    if (activeUploadPartitionName) {
      if (selectedId === 'all' || selectedId === 'default') {
        activeUploadPartitionName.textContent = 'Default';
      } else {
        activeUploadPartitionName.textContent = currentPart.name || selectedId;
      }
    }

    // Update table header title
    if (docTableTitle) {
      docTableTitle.textContent = 'Danh sách Tài liệu';
    }
  }

  if (docPartitionSelect) {
    docPartitionSelect.addEventListener('change', () => {
      updatePartitionUI();
      loadDocuments();
    });
  }

  // Create Partition Modal/Prompt
  if (createPartitionBtn) {
    createPartitionBtn.addEventListener('click', async () => {
      const name = prompt('Nhập tên phân vùng tài liệu mới:');
      if (!name) return;
      const cleanName = name.trim();
      if (!cleanName) return;

      try {
        const res = await fetch('/api/partitions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: cleanName })
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Không thể tạo phân vùng');
        }
        const data = await res.json();
        showToast(data.message || `Đã tạo phân vùng "${cleanName}"!`, 'success');
        await loadPartitions();
        if (docPartitionSelect) {
          docPartitionSelect.value = cleanName;
          updatePartitionUI();
          loadDocuments();
        }
      } catch (err) {
        showToast(err.message, 'error');
      }
    });
  }

  // Delete Partition
  if (deletePartitionBtn) {
    deletePartitionBtn.addEventListener('click', async () => {
      const selectedId = docPartitionSelect ? docPartitionSelect.value : '';
      if (!selectedId || selectedId === 'all' || selectedId === 'default') {
        showToast('Không thể xóa phân vùng này', 'warning');
        return;
      }

      if (!confirm(`Bạn có chắc chắn muốn xóa phân vùng "${selectedId}"?\nToàn bộ tệp và vector trong phân vùng này sẽ bị xóa vĩnh viễn!`)) {
        return;
      }

      deletePartitionBtn.disabled = true;

      try {
        const res = await fetch(`/api/partitions/${encodeURIComponent(selectedId)}`, {
          method: 'DELETE'
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Không thể xóa phân vùng');
        }
        showToast(`Đã xóa thành công phân vùng "${selectedId}"!`, 'success', 3500);

        await loadPartitions();
        if (docPartitionSelect) {
          docPartitionSelect.value = 'all';
          updatePartitionUI();
        }
        loadDocuments();
        fetchStats();
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        deletePartitionBtn.disabled = false;
      }
    });
  }

  async function loadDocuments() {
    try {
      const selectedPart = docPartitionSelect ? docPartitionSelect.value : 'all';
      const url = selectedPart && selectedPart !== 'all'
        ? `/api/documents?partition=${encodeURIComponent(selectedPart)}`
        : '/api/documents';

      const res = await fetch(url);
      if (!res.ok) throw new Error('Không thể tải danh sách tài liệu');
      const data = await res.json();
      const docs = data.documents || [];

      if (navDocCount) navDocCount.textContent = docs.length;

      if (docs.length === 0) {
        documentsTableBody.innerHTML = `
          <tr>
            <td colspan="7" class="text-center py-4" style="color: var(--text-dim);">Chưa có tài liệu nào trong phân vùng này. Hãy kéo thả tải lên tệp đầu tiên!</td>
          </tr>
        `;
        return;
      }

      documentsTableBody.innerHTML = docs.map(doc => {
        const ext = doc.extension.replace('.', '');
        let badgeClass = 'file-type-badge';
        if (ext === 'pdf') badgeClass += ' pdf';
        else if (ext === 'docx') badgeClass += ' docx';
        else badgeClass += ' txt';

        const statusHtml = doc.is_indexed
          ? '<span style="color: #34d399; font-weight: 600; font-size: 0.78rem;">✓ Đã lập chỉ mục</span>'
          : '<span style="color: #f59e0b; font-weight: 600; font-size: 0.78rem;">Chưa nạp (Mới)</span>';

        const isDefault = !doc.partition || doc.partition.toLowerCase() === 'default';
        const partLabel = isDefault ? 'Default' : doc.partition;
        const partBadgeClass = isDefault ? 'badge-partition default' : 'badge-partition';

        return `
          <tr>
            <td style="font-weight: 600; color: white;">
              <span class="doc-file-cell">
                <svg class="file-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                </svg>
                ${escapeHtml(doc.name)}
              </span>
            </td>
            <td>
              <span class="${partBadgeClass}">
                <img src="/static/folder.png" class="table-folder-icon" alt="Folder">
                ${escapeHtml(partLabel)}
              </span>
            </td>
            <td><span class="${badgeClass}">${ext.toUpperCase()}</span></td>
            <td style="font-family: var(--font-mono);">${doc.size_kb} KB</td>
            <td style="color: var(--text-muted);">${doc.modified_time}</td>
            <td>${statusHtml}</td>
            <td style="text-align: center;">
              <button class="btn btn-danger btn-delete-doc" data-filename="${escapeHtml(doc.name)}" data-partition="${escapeHtml(doc.partition)}" title="Xóa tài liệu và cập nhật lại chỉ mục">
                ${TRASH_ICON_SVG}
              </button>
            </td>
          </tr>
        `;
      }).join('');
    } catch (err) {
      console.error(err);
      documentsTableBody.innerHTML = `
        <tr><td colspan="7" class="text-center py-4" style="color: #f87171;">Lỗi tải tài liệu: ${escapeHtml(err.message)}</td></tr>
      `;
    }
  }

  refreshDocsBtn.addEventListener('click', () => {
    loadPartitions();
    loadDocuments();
  });

  // Event delegation for deleting document
  documentsTableBody.addEventListener('click', async (e) => {
    const deleteBtn = e.target.closest('.btn-delete-doc');
    if (!deleteBtn) return;

    const filename = deleteBtn.getAttribute('data-filename');
    const partition = deleteBtn.getAttribute('data-partition') || 'default';
    if (!filename) return;

    if (!confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}" (Phân vùng: ${partition})?\nHệ thống sẽ tự động cập nhật lại chỉ mục vector.`)) {
      return;
    }

    deleteBtn.disabled = true;

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}?partition=${encodeURIComponent(partition)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Không thể xóa tài liệu');
      }
      showToast(`Đã xóa thành công tài liệu "${filename}"!`, 'success', 3500);
      await loadPartitions();
      loadDocuments();
      fetchStats();
    } catch (err) {
      showToast(err.message, 'error');
      deleteBtn.disabled = false;
    }
  });

  // File Upload via Dropzone
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFilesUpload(fileInput.files);
    }
  });

  async function handleFilesUpload(filesList) {
    const selectedPart = docPartitionSelect ? docPartitionSelect.value : 'default';
    const actualPart = (selectedPart === 'all' || !selectedPart) ? 'default' : selectedPart;
    const displayPart = actualPart.toLowerCase() === 'default' ? 'Default' : actualPart;

    const formData = new FormData();
    for (let i = 0; i < filesList.length; i++) {
      formData.append('files', filesList[i]);
    }
    formData.append('partition', actualPart);

    showToast(`Đang tải lên và tự động lập chỉ mục ${filesList.length} tệp vào phân vùng [${displayPart}]...`, 'info', 4000);

    try {
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Tải lên thất bại');
      }
      const data = await res.json();

      if (data.new_chunks > 0) {
        showToast(`✅ Đã tải lên và lập chỉ mục ${data.new_chunks} chunks mới vào phân vùng [${displayPart}]!`, 'success', 5000);
      } else {
        showToast(`✅ Đã tải lên ${data.uploaded_files.length} tệp thành công!`, 'success', 4000);
      }

      await loadPartitions();
      loadDocuments();
      fetchStats();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  // Full Re-index Action (Tái tạo toàn bộ)
  rebuildIndexBtn.addEventListener('click', async () => {
    rebuildIndexBtn.disabled = true;
    showToast('Đang tái tạo toàn bộ HNSW Index (Full Rebuild)...', 'info', 6000);

    try {
      const res = await fetch('/api/index/rebuild', { method: 'POST' });
      if (!res.ok) throw new Error('Không thể tái tạo index');
      const data = await res.json();

      showToast(`✅ Tái tạo thành công: ${data.chunks_indexed} chunks trong ${data.duration_seconds}s!`, 'success', 5000);
      await loadPartitions();
      loadDocuments();
      fetchStats();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      rebuildIndexBtn.disabled = false;
    }
  });

  // -------------------------------------------------------------
  // 6. Parameter Sliders & Config (Only Chunk Size & Overlap)
  // -------------------------------------------------------------
  if (cfgChunkSize && valChunkSize) {
    cfgChunkSize.addEventListener('input', () => {
      valChunkSize.textContent = `${cfgChunkSize.value} ký tự`;
    });
  }
  if (cfgChunkOverlap && valChunkOverlap) {
    cfgChunkOverlap.addEventListener('input', () => {
      valChunkOverlap.textContent = `${cfgChunkOverlap.value} ký tự`;
    });
  }

  if (saveConfigBtn) {
    saveConfigBtn.addEventListener('click', async () => {
      saveConfigBtn.disabled = true;
      try {
        const payload = {
          chunk_size: parseInt(cfgChunkSize.value, 10),
          chunk_overlap: parseInt(cfgChunkOverlap.value, 10),
          top_k: 3,
          similarity_threshold: 0.50,
          vector_top_k: 15,
          bm25_top_k: 15,
          rrf_top_n: 10,
          model_name: 'gemini-3.5-flash-lite',
        };

        const res = await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Không thể lưu cấu hình');
        showToast('Đã lưu cấu hình tham số thành công!', 'success');
        fetchStats();
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        saveConfigBtn.disabled = false;
      }
    });
  }

  // Helper
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initial Load
  loadPartitions();
  loadDocuments();
  fetchStats();
});
