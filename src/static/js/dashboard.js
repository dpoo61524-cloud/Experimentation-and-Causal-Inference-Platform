/**
 * Experimentation and Causal Inference Platform - Dashboard Client Logic
 * Handles real API health checks, streaming CSV upload, benchmark dataset loading,
 * and schema inspector rendering.
 */

document.addEventListener('DOMContentLoaded', function () {
    // --- Elements ---
    const healthBadge = document.getElementById('health-badge');
    const healthText = document.getElementById('health-text');

    const fileInput = document.getElementById('csv-file-input');
    const fileMetaDisplay = document.getElementById('file-meta-display');
    const uploadBtn = document.getElementById('upload-csv-btn');
    const loadBenchmarkBtn = document.getElementById('load-benchmark-btn');

    const statusBanner = document.getElementById('status-banner');
    const emptyDatasetState = document.getElementById('empty-dataset-state');
    const loadedDatasetView = document.getElementById('loaded-dataset-view');

    // KPI Elements
    const kpiFilename = document.getElementById('kpi-filename');
    const kpiRowCount = document.getElementById('kpi-row-count');
    const kpiColumnCount = document.getElementById('kpi-column-count');
    const kpiFileSize = document.getElementById('kpi-file-size');
    const kpiMissingTotal = document.getElementById('kpi-missing-total');

    // Tables
    const previewTableCaption = document.getElementById('preview-table-caption');
    const previewTableHeader = document.getElementById('preview-table-header');
    const previewTableBody = document.getElementById('preview-table-body');
    const schemaTableBody = document.getElementById('schema-table-body');

    // Section 2: Causal Analysis Elements
    const ateStatusBanner = document.getElementById('ate-status-banner');
    const ateUnloadedNotice = document.getElementById('ate-unloaded-notice');
    const ateFormContainer = document.getElementById('ate-form-container');
    const ateDatasetName = document.getElementById('ate-dataset-name');
    const ateDatasetId = document.getElementById('ate-dataset-id');
    const ateDatasetRows = document.getElementById('ate-dataset-rows');
    const ateDatasetCols = document.getElementById('ate-dataset-cols');

    const ateForm = document.getElementById('ate-analysis-form');
    const ateTreatmentSelect = document.getElementById('ate-treatment-select');
    const ateOutcomeSelect = document.getElementById('ate-outcome-select');
    const ateTreatedValInput = document.getElementById('ate-treated-val-input');
    const ateUntreatedValInput = document.getElementById('ate-untreated-val-input');
    const ateCiLevelInput = document.getElementById('ate-ci-level-input');
    const ateDropMissingCheckbox = document.getElementById('ate-drop-missing-checkbox');
    const runAteBtn = document.getElementById('run-ate-btn');

    // Section 2: Results Elements
    const ateResultsPanel = document.getElementById('ate-results-panel');
    const resDatasetName = document.getElementById('res-dataset-name');
    const resTotalRows = document.getElementById('res-total-rows');
    const resEvalRows = document.getElementById('res-eval-rows');
    const resDroppedRows = document.getElementById('res-dropped-rows');
    const resAtePoint = document.getElementById('res-ate-point');
    const resAteSe = document.getElementById('res-ate-se');
    const resAteCi = document.getElementById('res-ate-ci');
    const resAteCiSub = document.getElementById('res-ate-ci-sub');
    const resMeanTreated = document.getElementById('res-mean-treated');
    const resCountTreated = document.getElementById('res-count-treated');
    const resMeanControl = document.getElementById('res-mean-control');
    const resCountControl = document.getElementById('res-count-control');
    const resAteDf = document.getElementById('res-ate-df');
    const resInterpretationText = document.getElementById('res-interpretation-text');
    const resAssumptionsText = document.getElementById('res-assumptions-text');

    // Section 3: Causal Analysis (Regression-Adjusted ATE) Elements
    const regStatusBanner = document.getElementById('reg-status-banner');
    const regUnloadedNotice = document.getElementById('reg-unloaded-notice');
    const regFormContainer = document.getElementById('reg-form-container');
    const regDatasetName = document.getElementById('reg-dataset-name');
    const regDatasetId = document.getElementById('reg-dataset-id');
    const regDatasetRows = document.getElementById('reg-dataset-rows');
    const regDatasetCols = document.getElementById('reg-dataset-cols');

    const regForm = document.getElementById('reg-analysis-form');
    const regTreatmentSelect = document.getElementById('reg-treatment-select');
    const regOutcomeSelect = document.getElementById('reg-outcome-select');
    const regConfoundersContainer = document.getElementById('reg-confounders-container');
    const regTreatedValInput = document.getElementById('reg-treated-val-input');
    const regUntreatedValInput = document.getElementById('reg-untreated-val-input');
    const regDropMissingCheckbox = document.getElementById('reg-drop-missing-checkbox');
    const runRegBtn = document.getElementById('run-reg-btn');

    // Section 3: Results Elements
    const regResultsPanel = document.getElementById('reg-results-panel');
    const resRegDatasetName = document.getElementById('res-reg-dataset-name');
    const resRegTotalRows = document.getElementById('res-reg-total-rows');
    const resRegEvalRows = document.getElementById('res-reg-eval-rows');
    const resRegDroppedRows = document.getElementById('res-reg-dropped-rows');
    const resRegAdjustedPoint = document.getElementById('res-reg-adjusted-point');
    const resRegRawPoint = document.getElementById('res-reg-raw-point');
    const resRegR2 = document.getElementById('res-reg-r2');
    const resRegMeanTreated = document.getElementById('res-reg-mean-treated');
    const resRegMeanControl = document.getElementById('res-reg-mean-control');
    const resRegFeatureCount = document.getElementById('res-reg-feature-count');
    const resRegNumConf = document.getElementById('res-reg-num-conf');
    const resRegCatConf = document.getElementById('res-reg-cat-conf');
    const resRegFeaturesList = document.getElementById('res-reg-features-list');
    const resRegInterpretationText = document.getElementById('res-reg-interpretation-text');
    const resRegAssumptionsText = document.getElementById('res-reg-assumptions-text');

    // Section 4: Causal Diagnostics (Propensity Score Estimation & Overlap) Elements
    const psStatusBanner = document.getElementById('ps-status-banner');
    const psUnloadedNotice = document.getElementById('ps-unloaded-notice');
    const psFormContainer = document.getElementById('ps-form-container');
    const psDatasetName = document.getElementById('ps-dataset-name');
    const psDatasetId = document.getElementById('ps-dataset-id');
    const psDatasetRows = document.getElementById('ps-dataset-rows');
    const psDatasetCols = document.getElementById('ps-dataset-cols');

    const psForm = document.getElementById('ps-analysis-form');
    const psTreatmentSelect = document.getElementById('ps-treatment-select');
    const psNearZeroInput = document.getElementById('ps-near-zero-input');
    const psNearOneInput = document.getElementById('ps-near-one-input');
    const psConfoundersContainer = document.getElementById('ps-confounders-container');
    const psTreatedValInput = document.getElementById('ps-treated-val-input');
    const psUntreatedValInput = document.getElementById('ps-untreated-val-input');
    const psDropMissingCheckbox = document.getElementById('ps-drop-missing-checkbox');
    const runPsBtn = document.getElementById('run-ps-btn');

    // Section 4: Results Elements
    const psResultsPanel = document.getElementById('ps-results-panel');
    const resPsDatasetName = document.getElementById('res-ps-dataset-name');
    const resPsTotalRows = document.getElementById('res-ps-total-rows');
    const resPsEvalRows = document.getElementById('res-ps-eval-rows');
    const resPsDroppedRows = document.getElementById('res-ps-dropped-rows');
    const resPsOverlapStatus = document.getElementById('res-ps-overlap-status');
    const resPsOverlapSub = document.getElementById('res-ps-overlap-sub');
    const resPsSupportRange = document.getElementById('res-ps-support-range');
    const resPsTreatedRange = document.getElementById('res-ps-treated-range');
    const resPsTreatedCountSub = document.getElementById('res-ps-treated-count-sub');
    const resPsControlRange = document.getElementById('res-ps-control-range');
    const resPsControlCountSub = document.getElementById('res-ps-control-count-sub');
    const resPsBoundaryCount = document.getElementById('res-ps-boundary-count');
    const resPsBoundarySub = document.getElementById('res-ps-boundary-sub');
    const resPsPseudoR2 = document.getElementById('res-ps-pseudo-r2');
    const resPsConvergedSub = document.getElementById('res-ps-converged-sub');
    const psBalanceTbody = document.getElementById('ps-balance-tbody');
    const resPsModelType = document.getElementById('res-ps-model-type');
    const resPsNumConf = document.getElementById('res-ps-num-conf');
    const resPsCatConf = document.getElementById('res-ps-cat-conf');
    const resPsFeaturesList = document.getElementById('res-ps-features-list');
    const resPsInterpretationText = document.getElementById('res-ps-interpretation-text');
    const resPsAssumptionsText = document.getElementById('res-ps-assumptions-text');

    // Section 5: Causal Analysis (IPTW ATE) Elements
    const iptwStatusBanner = document.getElementById('iptw-status-banner');
    const iptwUnloadedNotice = document.getElementById('iptw-unloaded-notice');
    const iptwFormContainer = document.getElementById('iptw-form-container');
    const iptwDatasetName = document.getElementById('iptw-dataset-name');
    const iptwDatasetId = document.getElementById('iptw-dataset-id');
    const iptwDatasetRows = document.getElementById('iptw-dataset-rows');
    const iptwDatasetCols = document.getElementById('iptw-dataset-cols');

    const iptwForm = document.getElementById('iptw-analysis-form');
    const iptwTreatmentSelect = document.getElementById('iptw-treatment-select');
    const iptwOutcomeSelect = document.getElementById('iptw-outcome-select');
    const iptwConfoundersContainer = document.getElementById('iptw-confounders-container');
    const iptwExtremeWeightThresholdInput = document.getElementById('iptw-extreme-weight-threshold-input');
    const iptwMaxWeightInput = document.getElementById('iptw-max-weight-input');
    const iptwTruncateCheckbox = document.getElementById('iptw-truncate-checkbox');
    const iptwTreatedValInput = document.getElementById('iptw-treated-val-input');
    const iptwUntreatedValInput = document.getElementById('iptw-untreated-val-input');
    const iptwDropMissingCheckbox = document.getElementById('iptw-drop-missing-checkbox');
    const runIptwBtn = document.getElementById('run-iptw-btn');

    // Section 5: Results Elements
    const iptwResultsPanel = document.getElementById('iptw-results-panel');
    const resIptwDatasetName = document.getElementById('res-iptw-dataset-name');
    const resIptwTotalRows = document.getElementById('res-iptw-total-rows');
    const resIptwEvalRows = document.getElementById('res-iptw-eval-rows');
    const resIptwDroppedRows = document.getElementById('res-iptw-dropped-rows');
    const resIptwAte = document.getElementById('res-iptw-ate');
    const resIptwUnadjustedAte = document.getElementById('res-iptw-unadjusted-ate');
    const resIptwMeanTreated = document.getElementById('res-iptw-mean-treated');
    const resIptwTreatedSub = document.getElementById('res-iptw-treated-sub');
    const resIptwMeanControl = document.getElementById('res-iptw-mean-control');
    const resIptwControlSub = document.getElementById('res-iptw-control-sub');
    const resIptwEssTotal = document.getElementById('res-iptw-ess-total');
    const resIptwEssSub = document.getElementById('res-iptw-ess-sub');
    const resIptwWeightSummary = document.getElementById('res-iptw-weight-summary');
    const resIptwWeightSub = document.getElementById('res-iptw-weight-sub');
    const iptwBalanceTbody = document.getElementById('iptw-balance-tbody');
    const iptwWarningsCard = document.getElementById('iptw-warnings-card');
    const iptwWarningsList = document.getElementById('iptw-warnings-list');
    const resIptwInterpretationText = document.getElementById('res-iptw-interpretation-text');
    const resIptwAssumptionsText = document.getElementById('res-iptw-assumptions-text');

    // Active session dataset metadata
    let currentDataset = null;

    // --- Format Helpers ---
    function formatBytes(bytes) {
        if (!bytes || bytes <= 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function formatNumber(num) {
        if (num === undefined || num === null) return '0';
        return Number(num).toLocaleString();
    }

    function showStatus(message, type) {
        statusBanner.className = 'status-indicator status-visible status-' + type;
        statusBanner.textContent = message;
    }

    function clearStatus() {
        statusBanner.className = 'status-indicator';
        statusBanner.textContent = '';
    }

    function setBusy(isBusy) {
        if (uploadBtn) uploadBtn.disabled = isBusy;
        if (loadBenchmarkBtn) loadBenchmarkBtn.disabled = isBusy;
        if (fileInput) fileInput.disabled = isBusy;
    }

    // --- 1. Real API Health Check ---
    function checkHealth() {
        fetch('/health', { method: 'GET', headers: { 'Accept': 'application/json' } })
            .then(function (res) {
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
            })
            .then(function (data) {
                if (data && data.status === 'healthy') {
                    healthBadge.className = 'health-badge health-healthy';
                    healthText.textContent = 'API: Healthy';
                } else {
                    healthBadge.className = 'health-badge health-unhealthy';
                    healthText.textContent = 'API: Unhealthy';
                }
            })
            .catch(function () {
                healthBadge.className = 'health-badge health-unhealthy';
                healthText.textContent = 'API: Offline';
            });
    }

    checkHealth();

    // --- 2. File Selection Handler ---
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            clearStatus();
            const file = fileInput.files[0];
            if (file) {
                fileMetaDisplay.textContent = 'Selected: ' + file.name + ' (' + formatBytes(file.size) + ')';
                uploadBtn.disabled = false;
            } else {
                fileMetaDisplay.textContent = 'No file chosen (Max upload limit: 500 MB)';
                uploadBtn.disabled = true;
            }
        });
    }

    // --- 3. CSV Upload (Streaming Request Body) ---
    if (uploadBtn) {
        uploadBtn.addEventListener('click', function () {
            const file = fileInput.files[0];
            if (!file) {
                showStatus('Please select a CSV file to upload.', 'error');
                return;
            }

            setBusy(true);
            showStatus('Streaming and inspecting CSV dataset on server... Please wait.', 'info');

            fetch('/api/dataset/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/csv',
                    'X-Filename': encodeURIComponent(file.name),
                },
                body: file,
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'Upload failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (data) {
                    currentDataset = data;
                    renderDatasetView(data);
                    showStatus('Dataset "' + data.filename + '" successfully ingested and verified (' + formatNumber(data.row_count) + ' rows).', 'success');
                    setBusy(false);
                })
                .catch(function (err) {
                    showStatus('Error: ' + err.message, 'error');
                    setBusy(false);
                });
        });
    }

    // --- 4. Load Hillstrom Benchmark Dataset ---
    if (loadBenchmarkBtn) {
        loadBenchmarkBtn.addEventListener('click', function () {
            setBusy(true);
            showStatus('Loading Hillstrom email marketing benchmark dataset (64,000 rows)... Please wait.', 'info');

            fetch('/api/dataset/load-benchmark', {
                method: 'POST',
                headers: { 'Accept': 'application/json' },
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'Benchmark load failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (data) {
                    currentDataset = data;
                    renderDatasetView(data);
                    showStatus('Hillstrom benchmark dataset successfully loaded (' + formatNumber(data.row_count) + ' rows, 12 columns).', 'success');
                    setBusy(false);
                })
                .catch(function (err) {
                    showStatus('Benchmark Error: ' + err.message, 'error');
                    setBusy(false);
                });
        });
    }

    // --- 5. Render Dataset Overview, Preview Table & Schema Inspector ---
    function renderDatasetView(data) {
        if (!data) return;

        // Toggle states
        if (emptyDatasetState) emptyDatasetState.style.display = 'none';
        if (loadedDatasetView) loadedDatasetView.style.display = 'block';

        // 1. Overview KPI Cards
        const benchmarkTag = data.is_benchmark ? ' [Benchmark]' : '';
        kpiFilename.textContent = data.filename + benchmarkTag;
        kpiRowCount.textContent = formatNumber(data.row_count);
        kpiColumnCount.textContent = formatNumber(data.column_count);
        kpiFileSize.textContent = formatBytes(data.file_size_bytes);

        // Sum missing values across all columns
        let totalMissing = 0;
        if (Array.isArray(data.columns)) {
            data.columns.forEach(function (c) {
                totalMissing += c.missing_count || 0;
            });
        }
        kpiMissingTotal.textContent = formatNumber(totalMissing);

        // 2. Bounded 5-Row Preview Table
        const previewRows = data.preview_rows || [];
        const columns = data.columns || [];

        previewTableCaption.textContent = 'Preview: Showing first ' + previewRows.length + ' of ' + formatNumber(data.row_count) + ' records. Full dataset preserved for analysis.';

        // Render Headers
        previewTableHeader.innerHTML = '';
        const trHead = document.createElement('tr');
        columns.forEach(function (col) {
            const th = document.createElement('th');
            th.textContent = col.name;
            trHead.appendChild(th);
        });
        previewTableHeader.appendChild(trHead);

        // Render Rows
        previewTableBody.innerHTML = '';
        if (previewRows.length === 0) {
            const trEmpty = document.createElement('tr');
            const tdEmpty = document.createElement('td');
            tdEmpty.colSpan = columns.length || 1;
            tdEmpty.textContent = 'No preview rows available (header-only dataset).';
            tdEmpty.style.textAlign = 'center';
            trEmpty.appendChild(tdEmpty);
            previewTableBody.appendChild(trEmpty);
        } else {
            previewRows.forEach(function (row) {
                const tr = document.createElement('tr');
                columns.forEach(function (col) {
                    const td = document.createElement('td');
                    const val = row[col.name];
                    td.textContent = (val === null || val === undefined || val === '') ? '—' : String(val);
                    tr.appendChild(td);
                });
                previewTableBody.appendChild(tr);
            });
        }

        // 3. Schema Inspector Table
        schemaTableBody.innerHTML = '';
        columns.forEach(function (col) {
            const tr = document.createElement('tr');

            // Column Name
            const tdName = document.createElement('td');
            tdName.style.fontWeight = '600';
            tdName.textContent = col.name;
            tr.appendChild(tdName);

            // Inferred Type Badge
            const tdType = document.createElement('td');
            const spanType = document.createElement('span');
            spanType.className = 'type-badge type-' + (col.inferred_type || 'unknown').toLowerCase();
            spanType.textContent = col.inferred_type || 'unknown';
            tdType.appendChild(spanType);
            tr.appendChild(tdType);

            // Missing Count
            const tdMissing = document.createElement('td');
            tdMissing.textContent = formatNumber(col.missing_count);
            tr.appendChild(tdMissing);

            // Missing Percentage
            const tdPct = document.createElement('td');
            tdPct.textContent = Number(col.missing_percentage).toFixed(1) + '%';
            tr.appendChild(tdPct);

            // Valid Count
            const tdValid = document.createElement('td');
            tdValid.textContent = formatNumber(col.total_valid_count);
            tr.appendChild(tdValid);

            schemaTableBody.appendChild(tr);
        });

        // Initialize causal analysis workflows for this dataset
        setupAteAnalysis(data);
        setupRegAnalysis(data);
        setupPsAnalysis(data);
        setupIptwAnalysis(data);
    }

    // --- 6. Causal Analysis (Unadjusted ATE) Logic ---
    function showAteStatus(message, type) {
        if (!ateStatusBanner) return;
        ateStatusBanner.className = 'status-indicator status-visible status-' + type;
        ateStatusBanner.textContent = message;
    }

    function clearAteStatus() {
        if (!ateStatusBanner) return;
        ateStatusBanner.className = 'status-indicator';
        ateStatusBanner.textContent = '';
    }

    function setupAteAnalysis(data) {
        if (!data || !data.columns) return;
        if (ateUnloadedNotice) ateUnloadedNotice.style.display = 'none';
        if (ateFormContainer) ateFormContainer.style.display = 'block';
        if (ateResultsPanel) ateResultsPanel.style.display = 'none';
        clearAteStatus();

        if (ateDatasetName) ateDatasetName.textContent = data.filename + (data.is_benchmark ? ' [Benchmark]' : '');
        if (ateDatasetId) ateDatasetId.textContent = data.dataset_id;
        if (ateDatasetRows) ateDatasetRows.textContent = formatNumber(data.row_count);
        if (ateDatasetCols) ateDatasetCols.textContent = formatNumber(data.column_count);

        if (ateTreatmentSelect && ateOutcomeSelect) {
            ateTreatmentSelect.innerHTML = '<option value="">-- Choose treatment column --</option>';
            ateOutcomeSelect.innerHTML = '<option value="">-- Choose outcome column --</option>';

            let defaultTreatment = '';
            let defaultOutcome = '';

            data.columns.forEach(function (col) {
                const optT = document.createElement('option');
                optT.value = col.name;
                optT.textContent = col.name + ' (' + col.inferred_type + ')';
                ateTreatmentSelect.appendChild(optT);

                const optO = document.createElement('option');
                optO.value = col.name;
                optO.textContent = col.name + ' (' + col.inferred_type + ')';
                ateOutcomeSelect.appendChild(optO);

                const nameLower = col.name.toLowerCase();
                if (!defaultTreatment && (col.inferred_type === 'boolean' || nameLower === 'treatment' || nameLower === 'mens')) {
                    defaultTreatment = col.name;
                }
                if (!defaultOutcome && (col.inferred_type === 'numeric' || nameLower === 'spend' || nameLower === 'conversion' || nameLower === 'outcome')) {
                    defaultOutcome = col.name;
                }
            });

            if (defaultTreatment) ateTreatmentSelect.value = defaultTreatment;
            if (defaultOutcome && defaultOutcome !== defaultTreatment) {
                ateOutcomeSelect.value = defaultOutcome;
            }
        }

        validateAteForm();
    }

    function validateAteForm() {
        if (!currentDataset) {
            if (runAteBtn) runAteBtn.disabled = true;
            return false;
        }

        const tCol = ateTreatmentSelect ? ateTreatmentSelect.value : '';
        const oCol = ateOutcomeSelect ? ateOutcomeSelect.value : '';
        const ciVal = ateCiLevelInput ? parseFloat(ateCiLevelInput.value) : 0.95;

        if (!tCol || !oCol) {
            clearAteStatus();
            if (runAteBtn) runAteBtn.disabled = true;
            return false;
        }

        if (tCol === oCol) {
            showAteStatus('Treatment and Outcome columns cannot be identical ("' + tCol + '").', 'error');
            if (runAteBtn) runAteBtn.disabled = true;
            return false;
        }

        if (isNaN(ciVal) || ciVal <= 0 || ciVal >= 1) {
            showAteStatus('Confidence level must be strictly between 0 and 1 (exclusive), e.g. 0.95.', 'error');
            if (runAteBtn) runAteBtn.disabled = true;
            return false;
        }

        clearAteStatus();
        if (runAteBtn) runAteBtn.disabled = false;
        return true;
    }

    if (ateTreatmentSelect) ateTreatmentSelect.addEventListener('change', validateAteForm);
    if (ateOutcomeSelect) ateOutcomeSelect.addEventListener('change', validateAteForm);
    if (ateCiLevelInput) ateCiLevelInput.addEventListener('input', validateAteForm);

    if (runAteBtn) {
        runAteBtn.addEventListener('click', function (e) {
            e.preventDefault();
            if (!validateAteForm()) return;
            if (!currentDataset) {
                showAteStatus('No active dataset found. Please load or upload a dataset first.', 'error');
                return;
            }

            const payload = {
                treatment_col: ateTreatmentSelect.value,
                outcome_col: ateOutcomeSelect.value,
                drop_missing: ateDropMissingCheckbox ? ateDropMissingCheckbox.checked : false,
                confidence_level: parseFloat(ateCiLevelInput.value) || 0.95,
            };

            const treatedVal = ateTreatedValInput ? ateTreatedValInput.value.trim() : '';
            if (treatedVal !== '') payload.treated_value = treatedVal;

            const untreatedVal = ateUntreatedValInput ? ateUntreatedValInput.value.trim() : '';
            if (untreatedVal !== '') payload.untreated_value = untreatedVal;

            runAteBtn.disabled = true;
            if (ateResultsPanel) ateResultsPanel.style.display = 'none';
            showAteStatus('Estimating Unadjusted ATE on server (evaluating all records)... Please wait.', 'info');

            fetch('/api/dataset/' + encodeURIComponent(currentDataset.dataset_id) + '/causal/unadjusted-ate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(payload),
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'Analysis request failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (results) {
                    renderAteResults(results);
                    showAteStatus('Analysis complete: successfully evaluated ' + formatNumber(results.n_total) + ' observations from ' + results.dataset_filename + '.', 'success');
                })
                .catch(function (err) {
                    showAteStatus('Causal Estimation Error: ' + err.message, 'error');
                })
                .finally(function () {
                    runAteBtn.disabled = false;
                });
        });
    }

    function renderAteResults(r) {
        if (!r) return;
        if (ateResultsPanel) ateResultsPanel.style.display = 'block';

        if (resDatasetName) resDatasetName.textContent = r.dataset_filename || '—';
        if (resTotalRows) resTotalRows.textContent = formatNumber(r.total_dataset_rows);
        if (resEvalRows) resEvalRows.textContent = formatNumber(r.n_total);
        if (resDroppedRows) resDroppedRows.textContent = formatNumber(r.rows_dropped);

        if (resAtePoint) {
            const prefix = (r.ate > 0) ? '+' : '';
            resAtePoint.textContent = prefix + Number(r.ate).toFixed(4);
        }

        if (resAteSe) {
            resAteSe.textContent = (r.std_error !== null && r.std_error !== undefined) ? Number(r.std_error).toFixed(4) : 'Not estimated';
        }

        if (resAteCi) {
            if (r.ci_lower !== null && r.ci_upper !== null && r.ci_lower !== undefined && r.ci_upper !== undefined) {
                resAteCi.textContent = '[' + Number(r.ci_lower).toFixed(4) + ', ' + Number(r.ci_upper).toFixed(4) + ']';
            } else {
                resAteCi.textContent = 'Not available';
            }
        }

        if (resAteCiSub) {
            const confPct = Math.round((r.confidence_level || 0.95) * 100);
            resAteCiSub.textContent = confPct + '% Two-sided Welch interval';
        }

        if (resMeanTreated) {
            resMeanTreated.textContent = (r.mean_outcome_treated !== null && r.mean_outcome_treated !== undefined) ? Number(r.mean_outcome_treated).toFixed(4) : 'Not available';
        }
        if (resCountTreated) {
            const lbl = (r.treated_value !== null && r.treated_value !== undefined) ? ' (label: ' + r.treated_value + ')' : '';
            resCountTreated.textContent = 'N₁ = ' + formatNumber(r.n_treated) + lbl;
        }

        if (resMeanControl) {
            resMeanControl.textContent = (r.mean_outcome_control !== null && r.mean_outcome_control !== undefined) ? Number(r.mean_outcome_control).toFixed(4) : 'Not available';
        }
        if (resCountControl) {
            const lbl = (r.control_value !== null && r.control_value !== undefined) ? ' (label: ' + r.control_value + ')' : '';
            resCountControl.textContent = 'N₀ = ' + formatNumber(r.n_control) + lbl;
        }

        if (resAteDf) {
            resAteDf.textContent = (r.degrees_of_freedom !== null && r.degrees_of_freedom !== undefined) ? Number(r.degrees_of_freedom).toFixed(1) : 'Not available';
        }

        if (resInterpretationText) {
            resInterpretationText.textContent = r.interpretation || 'No interpretation provided.';
        }
        if (resAssumptionsText) {
            resAssumptionsText.textContent = r.assumptions || 'No identification assumptions documented.';
        }
    }

    // --- 7. Causal Analysis (Regression-Adjusted ATE) Logic (Stage 7.4B) ---
    function showRegStatus(message, type) {
        if (!regStatusBanner) return;
        regStatusBanner.className = 'status-indicator status-visible status-' + type;
        regStatusBanner.textContent = message;
    }

    function clearRegStatus() {
        if (!regStatusBanner) return;
        regStatusBanner.className = 'status-indicator';
        regStatusBanner.textContent = '';
    }

    function setupRegAnalysis(data) {
        if (!data || !data.columns) return;
        if (regUnloadedNotice) regUnloadedNotice.style.display = 'none';
        if (regFormContainer) regFormContainer.style.display = 'block';
        if (regResultsPanel) regResultsPanel.style.display = 'none';
        clearRegStatus();

        if (regDatasetName) regDatasetName.textContent = data.filename + (data.is_benchmark ? ' [Benchmark]' : '');
        if (regDatasetId) regDatasetId.textContent = data.dataset_id;
        if (regDatasetRows) regDatasetRows.textContent = formatNumber(data.row_count);
        if (regDatasetCols) regDatasetCols.textContent = formatNumber(data.column_count);

        if (regTreatmentSelect && regOutcomeSelect) {
            regTreatmentSelect.innerHTML = '<option value="">-- Choose treatment column --</option>';
            regOutcomeSelect.innerHTML = '<option value="">-- Choose outcome column --</option>';

            let defaultTreatment = '';
            let defaultOutcome = '';

            data.columns.forEach(function (col) {
                const optT = document.createElement('option');
                optT.value = col.name;
                optT.textContent = col.name + ' (' + col.inferred_type + ')';
                regTreatmentSelect.appendChild(optT);

                const optO = document.createElement('option');
                optO.value = col.name;
                optO.textContent = col.name + ' (' + col.inferred_type + ')';
                regOutcomeSelect.appendChild(optO);

                const nameLower = col.name.toLowerCase();
                if (!defaultTreatment && (col.inferred_type === 'boolean' || nameLower === 'treatment' || nameLower === 'mens')) {
                    defaultTreatment = col.name;
                }
                if (!defaultOutcome && (col.inferred_type === 'numeric' || nameLower === 'spend' || nameLower === 'conversion' || nameLower === 'outcome')) {
                    defaultOutcome = col.name;
                }
            });

            if (defaultTreatment) regTreatmentSelect.value = defaultTreatment;
            if (defaultOutcome && defaultOutcome !== defaultTreatment) {
                regOutcomeSelect.value = defaultOutcome;
            }
        }

        // Populate confounder checkboxes
        if (regConfoundersContainer) {
            regConfoundersContainer.innerHTML = '';
            data.columns.forEach(function (col) {
                const label = document.createElement('label');
                label.className = 'confounder-checkbox-item';
                label.setAttribute('data-col', col.name);

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'reg-confounder';
                checkbox.value = col.name;

                // Check default confounders for Hillstrom if not treatment or outcome
                const nameLower = col.name.toLowerCase();
                if (data.is_benchmark && (nameLower === 'recency' || nameLower === 'history')) {
                    checkbox.checked = true;
                }

                checkbox.addEventListener('change', function () {
                    validateRegForm();
                });

                const spanName = document.createElement('span');
                spanName.textContent = col.name;

                const spanType = document.createElement('span');
                spanType.className = 'col-type-tag';
                spanType.textContent = col.inferred_type;

                label.appendChild(checkbox);
                label.appendChild(spanName);
                label.appendChild(spanType);

                regConfoundersContainer.appendChild(label);
            });
        }

        updateConfoundersAvailability();
        validateRegForm();
    }

    function updateConfoundersAvailability() {
        if (!regConfoundersContainer) return;
        const tCol = regTreatmentSelect ? regTreatmentSelect.value : '';
        const oCol = regOutcomeSelect ? regOutcomeSelect.value : '';

        const items = regConfoundersContainer.querySelectorAll('.confounder-checkbox-item');
        items.forEach(function (item) {
            const col = item.getAttribute('data-col');
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (!checkbox) return;

            if (col && (col === tCol || col === oCol)) {
                checkbox.checked = false;
                checkbox.disabled = true;
                item.classList.add('disabled');
            } else {
                checkbox.disabled = false;
                item.classList.remove('disabled');
            }
        });
    }

    function getSelectedConfounders() {
        if (!regConfoundersContainer) return [];
        const checkboxes = regConfoundersContainer.querySelectorAll('input[name="reg-confounder"]:checked');
        const selected = [];
        checkboxes.forEach(function (cb) {
            if (!cb.disabled && cb.value) {
                selected.push(cb.value);
            }
        });
        return selected;
    }

    function validateRegForm() {
        if (!currentDataset) {
            if (runRegBtn) runRegBtn.disabled = true;
            return false;
        }

        const tCol = regTreatmentSelect ? regTreatmentSelect.value : '';
        const oCol = regOutcomeSelect ? regOutcomeSelect.value : '';
        const selectedConfounders = getSelectedConfounders();

        if (!tCol || !oCol) {
            clearRegStatus();
            if (runRegBtn) runRegBtn.disabled = true;
            return false;
        }

        if (tCol === oCol) {
            showRegStatus('Treatment and Outcome columns cannot be identical ("' + tCol + '").', 'error');
            if (runRegBtn) runRegBtn.disabled = true;
            return false;
        }

        if (selectedConfounders.length === 0) {
            clearRegStatus();
            if (runRegBtn) runRegBtn.disabled = true;
            return false;
        }

        // Check role conflicts
        for (let i = 0; i < selectedConfounders.length; i++) {
            const cCol = selectedConfounders[i];
            if (cCol === tCol) {
                showRegStatus('Confounder "' + cCol + '" cannot also be the Treatment column.', 'error');
                if (runRegBtn) runRegBtn.disabled = true;
                return false;
            }
            if (cCol === oCol) {
                showRegStatus('Confounder "' + cCol + '" cannot also be the Outcome column.', 'error');
                if (runRegBtn) runRegBtn.disabled = true;
                return false;
            }
        }

        clearRegStatus();
        if (runRegBtn) runRegBtn.disabled = false;
        return true;
    }

    if (regTreatmentSelect) {
        regTreatmentSelect.addEventListener('change', function () {
            updateConfoundersAvailability();
            validateRegForm();
        });
    }

    if (regOutcomeSelect) {
        regOutcomeSelect.addEventListener('change', function () {
            updateConfoundersAvailability();
            validateRegForm();
        });
    }

    if (runRegBtn) {
        runRegBtn.addEventListener('click', function (e) {
            e.preventDefault();
            if (!validateRegForm()) return;
            if (!currentDataset) {
                showRegStatus('No active dataset found. Please load or upload a dataset first.', 'error');
                return;
            }

            const selectedConfounders = getSelectedConfounders();
            const payload = {
                treatment_col: regTreatmentSelect.value,
                outcome_col: regOutcomeSelect.value,
                confounder_cols: selectedConfounders,
                drop_missing: regDropMissingCheckbox ? regDropMissingCheckbox.checked : false,
            };

            const treatedVal = regTreatedValInput ? regTreatedValInput.value.trim() : '';
            if (treatedVal !== '') payload.treated_value = treatedVal;

            const untreatedVal = regUntreatedValInput ? regUntreatedValInput.value.trim() : '';
            if (untreatedVal !== '') payload.untreated_value = untreatedVal;

            runRegBtn.disabled = true;
            if (regResultsPanel) regResultsPanel.style.display = 'none';
            showRegStatus('Estimating Regression-Adjusted ATE on server (evaluating all records via OLS & G-computation)... Please wait.', 'info');

            fetch('/api/dataset/' + encodeURIComponent(currentDataset.dataset_id) + '/causal/regression-adjusted-ate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(payload),
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'Analysis request failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (results) {
                    renderRegResults(results);
                    showRegStatus('Analysis complete: successfully evaluated ' + formatNumber(results.n_total) + ' observations from ' + results.dataset_filename + '.', 'success');
                })
                .catch(function (err) {
                    showRegStatus('Causal Estimation Error: ' + err.message, 'error');
                })
                .finally(function () {
                    runRegBtn.disabled = false;
                });
        });
    }

    function renderRegResults(r) {
        if (!r) return;
        if (regResultsPanel) regResultsPanel.style.display = 'block';

        if (resRegDatasetName) resRegDatasetName.textContent = r.dataset_filename || '—';
        if (resRegTotalRows) resRegTotalRows.textContent = formatNumber(r.total_dataset_rows);
        if (resRegEvalRows) resRegEvalRows.textContent = formatNumber(r.n_total);
        if (resRegDroppedRows) resRegDroppedRows.textContent = formatNumber(r.rows_dropped);

        if (resRegAdjustedPoint) {
            const prefix = (r.adjusted_ate > 0) ? '+' : '';
            resRegAdjustedPoint.textContent = prefix + Number(r.adjusted_ate).toFixed(4);
        }

        if (resRegRawPoint) {
            const rawVal = (r.unadjusted_ate !== null && r.unadjusted_ate !== undefined) ? r.unadjusted_ate : r.raw_difference;
            const prefix = (rawVal > 0) ? '+' : '';
            resRegRawPoint.textContent = prefix + Number(rawVal).toFixed(4);
        }

        if (resRegR2) {
            resRegR2.textContent = (r.r_squared !== null && r.r_squared !== undefined) ? Number(r.r_squared).toFixed(4) : 'N/A';
        }

        if (resRegMeanTreated) {
            resRegMeanTreated.textContent = (r.mean_predicted_treated !== null && r.mean_predicted_treated !== undefined) ? Number(r.mean_predicted_treated).toFixed(4) : 'Not available';
        }

        if (resRegMeanControl) {
            resRegMeanControl.textContent = (r.mean_predicted_control !== null && r.mean_predicted_control !== undefined) ? Number(r.mean_predicted_control).toFixed(4) : 'Not available';
        }

        if (resRegFeatureCount) {
            resRegFeatureCount.textContent = (r.model_features && Array.isArray(r.model_features)) ? formatNumber(r.model_features.length) : '0';
        }

        if (resRegNumConf) {
            resRegNumConf.textContent = (r.numeric_confounders && r.numeric_confounders.length) ? r.numeric_confounders.join(', ') : 'None';
        }

        if (resRegCatConf) {
            resRegCatConf.textContent = (r.categorical_confounders && r.categorical_confounders.length) ? r.categorical_confounders.join(', ') : 'None';
        }

        if (resRegFeaturesList) {
            resRegFeaturesList.textContent = (r.model_features && r.model_features.length) ? r.model_features.join(', ') : 'None';
        }

        if (resRegInterpretationText) {
            resRegInterpretationText.textContent = r.interpretation || 'No interpretation provided.';
        }

        if (resRegAssumptionsText) {
            resRegAssumptionsText.textContent = r.assumptions || 'No identification assumptions documented.';
        }
    }

    // --- 8. Causal Diagnostics (Propensity Score Estimation) Logic (Stage 7.5A) ---
    function showPsStatus(message, type) {
        if (!psStatusBanner) return;
        psStatusBanner.className = 'status-indicator status-visible status-' + type;
        psStatusBanner.textContent = message;
    }

    function clearPsStatus() {
        if (!psStatusBanner) return;
        psStatusBanner.className = 'status-indicator';
        psStatusBanner.textContent = '';
    }

    function setupPsAnalysis(data) {
        if (!data || !data.columns) return;
        if (psUnloadedNotice) psUnloadedNotice.style.display = 'none';
        if (psFormContainer) psFormContainer.style.display = 'block';
        if (psResultsPanel) psResultsPanel.style.display = 'none';
        clearPsStatus();

        if (psDatasetName) psDatasetName.textContent = data.filename + (data.is_benchmark ? ' [Benchmark]' : '');
        if (psDatasetId) psDatasetId.textContent = data.dataset_id;
        if (psDatasetRows) psDatasetRows.textContent = formatNumber(data.row_count);
        if (psDatasetCols) psDatasetCols.textContent = formatNumber(data.column_count);

        if (psTreatmentSelect) {
            psTreatmentSelect.innerHTML = '<option value="">-- Choose treatment column --</option>';

            let defaultTreatment = '';

            data.columns.forEach(function (col) {
                const optT = document.createElement('option');
                optT.value = col.name;
                optT.textContent = col.name + ' (' + col.inferred_type + ')';
                psTreatmentSelect.appendChild(optT);

                const nameLower = col.name.toLowerCase();
                if (!defaultTreatment && (col.inferred_type === 'boolean' || nameLower === 'treatment' || nameLower === 'mens')) {
                    defaultTreatment = col.name;
                }
            });

            if (defaultTreatment) psTreatmentSelect.value = defaultTreatment;
        }

        // Populate confounder checkboxes
        if (psConfoundersContainer) {
            psConfoundersContainer.innerHTML = '';
            data.columns.forEach(function (col) {
                const label = document.createElement('label');
                label.className = 'confounder-checkbox-item';
                label.setAttribute('data-col', col.name);

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'ps-confounder';
                checkbox.value = col.name;

                // Check default confounders for Hillstrom if not treatment
                const nameLower = col.name.toLowerCase();
                if (data.is_benchmark && (nameLower === 'recency' || nameLower === 'history')) {
                    checkbox.checked = true;
                }

                checkbox.addEventListener('change', function () {
                    validatePsForm();
                });

                const spanName = document.createElement('span');
                spanName.textContent = col.name;

                const spanType = document.createElement('span');
                spanType.className = 'col-type-tag';
                spanType.textContent = col.inferred_type;

                label.appendChild(checkbox);
                label.appendChild(spanName);
                label.appendChild(spanType);

                psConfoundersContainer.appendChild(label);
            });
        }

        updatePsConfoundersAvailability();
        validatePsForm();
    }

    function updatePsConfoundersAvailability() {
        if (!psConfoundersContainer) return;
        const tCol = psTreatmentSelect ? psTreatmentSelect.value : '';

        const items = psConfoundersContainer.querySelectorAll('.confounder-checkbox-item');
        items.forEach(function (item) {
            const col = item.getAttribute('data-col');
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (!checkbox) return;

            if (col && col === tCol) {
                checkbox.checked = false;
                checkbox.disabled = true;
                item.classList.add('disabled');
            } else {
                checkbox.disabled = false;
                item.classList.remove('disabled');
            }
        });
    }

    function getSelectedPsConfounders() {
        if (!psConfoundersContainer) return [];
        const checkboxes = psConfoundersContainer.querySelectorAll('input[name="ps-confounder"]:checked');
        const selected = [];
        checkboxes.forEach(function (cb) {
            if (!cb.disabled && cb.value) {
                selected.push(cb.value);
            }
        });
        return selected;
    }

    function validatePsForm() {
        if (!currentDataset) {
            if (runPsBtn) runPsBtn.disabled = true;
            return false;
        }

        const tCol = psTreatmentSelect ? psTreatmentSelect.value : '';
        const selectedConfounders = getSelectedPsConfounders();

        if (!tCol) {
            clearPsStatus();
            if (runPsBtn) runPsBtn.disabled = true;
            return false;
        }

        if (selectedConfounders.length === 0) {
            clearPsStatus();
            if (runPsBtn) runPsBtn.disabled = true;
            return false;
        }

        // Check role conflicts
        for (let i = 0; i < selectedConfounders.length; i++) {
            const cCol = selectedConfounders[i];
            if (cCol === tCol) {
                showPsStatus('Confounder "' + cCol + '" cannot also be the Treatment column.', 'error');
                if (runPsBtn) runPsBtn.disabled = true;
                return false;
            }
        }

        // Validate thresholds
        const zVal = psNearZeroInput ? parseFloat(psNearZeroInput.value) : 0.05;
        const oVal = psNearOneInput ? parseFloat(psNearOneInput.value) : 0.95;
        if (isNaN(zVal) || isNaN(oVal) || zVal < 0 || oVal > 1 || zVal >= oVal) {
            showPsStatus('Near-zero threshold must be strictly less than near-one threshold (within [0, 1]).', 'error');
            if (runPsBtn) runPsBtn.disabled = true;
            return false;
        }

        clearPsStatus();
        if (runPsBtn) runPsBtn.disabled = false;
        return true;
    }

    if (psTreatmentSelect) {
        psTreatmentSelect.addEventListener('change', function () {
            updatePsConfoundersAvailability();
            validatePsForm();
        });
    }

    if (psNearZeroInput) psNearZeroInput.addEventListener('input', validatePsForm);
    if (psNearOneInput) psNearOneInput.addEventListener('input', validatePsForm);

    if (runPsBtn) {
        runPsBtn.addEventListener('click', function (e) {
            e.preventDefault();
            if (!validatePsForm()) return;
            if (!currentDataset) {
                showPsStatus('No active dataset found. Please load or upload a dataset first.', 'error');
                return;
            }

            const selectedConfounders = getSelectedPsConfounders();
            const payload = {
                treatment_col: psTreatmentSelect.value,
                confounder_cols: selectedConfounders,
                drop_missing: psDropMissingCheckbox ? psDropMissingCheckbox.checked : false,
                near_zero_threshold: psNearZeroInput ? parseFloat(psNearZeroInput.value) || 0.05 : 0.05,
                near_one_threshold: psNearOneInput ? parseFloat(psNearOneInput.value) || 0.95 : 0.95,
            };

            const treatedVal = psTreatedValInput ? psTreatedValInput.value.trim() : '';
            if (treatedVal !== '') payload.treated_value = treatedVal;

            const untreatedVal = psUntreatedValInput ? psUntreatedValInput.value.trim() : '';
            if (untreatedVal !== '') payload.untreated_value = untreatedVal;

            runPsBtn.disabled = true;
            if (psResultsPanel) psResultsPanel.style.display = 'none';
            showPsStatus('Estimating propensity scores via logistic regression (evaluating all records)... Please wait.', 'info');

            fetch('/api/dataset/' + encodeURIComponent(currentDataset.dataset_id) + '/causal/propensity-scores', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(payload),
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'Analysis request failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (results) {
                    renderPsResults(results);
                    showPsStatus('Propensity score estimation complete: modeled ' + formatNumber(results.n_total) + ' observations from ' + results.dataset_filename + '.', 'success');
                })
                .catch(function (err) {
                    showPsStatus('Propensity Score Estimation Error: ' + err.message, 'error');
                })
                .finally(function () {
                    runPsBtn.disabled = false;
                });
        });
    }

    function renderPsResults(r) {
        if (!r) return;
        if (psResultsPanel) psResultsPanel.style.display = 'block';

        if (resPsDatasetName) resPsDatasetName.textContent = r.dataset_filename || '—';
        if (resPsTotalRows) resPsTotalRows.textContent = formatNumber(r.total_dataset_rows);
        if (resPsEvalRows) resPsEvalRows.textContent = formatNumber(r.n_total);
        if (resPsDroppedRows) resPsDroppedRows.textContent = formatNumber(r.rows_dropped);

        const diag = r.overlap_diagnostics || {};

        if (resPsOverlapStatus) {
            if (diag.has_overlap) {
                resPsOverlapStatus.textContent = (diag.scores_near_zero > 0 || diag.scores_near_one > 0) ? 'Limited Overlap' : 'Common Support';
                resPsOverlapStatus.style.color = (diag.scores_near_zero > 0 || diag.scores_near_one > 0) ? '#D97706' : 'var(--deep-sage)';
            } else {
                resPsOverlapStatus.textContent = 'Violated (Disjoint)';
                resPsOverlapStatus.style.color = '#DC2626';
            }
        }

        if (resPsOverlapSub) {
            resPsOverlapSub.textContent = diag.has_overlap ? 'Positivity condition satisfied' : 'No common support between groups';
        }

        if (resPsSupportRange) {
            if (diag.common_support_range && Array.isArray(diag.common_support_range)) {
                resPsSupportRange.textContent = '[' + Number(diag.common_support_range[0]).toFixed(4) + ', ' + Number(diag.common_support_range[1]).toFixed(4) + ']';
            } else {
                resPsSupportRange.textContent = 'None (Disjoint)';
            }
        }

        if (resPsTreatedRange) {
            if (diag.treated_range && Array.isArray(diag.treated_range)) {
                resPsTreatedRange.textContent = '[' + Number(diag.treated_range[0]).toFixed(4) + ', ' + Number(diag.treated_range[1]).toFixed(4) + ']';
            } else {
                resPsTreatedRange.textContent = '—';
            }
        }
        if (resPsTreatedCountSub) {
            resPsTreatedCountSub.textContent = 'N₁ = ' + formatNumber(r.n_treated) + ' treated units';
        }

        if (resPsControlRange) {
            if (diag.control_range && Array.isArray(diag.control_range)) {
                resPsControlRange.textContent = '[' + Number(diag.control_range[0]).toFixed(4) + ', ' + Number(diag.control_range[1]).toFixed(4) + ']';
            } else {
                resPsControlRange.textContent = '—';
            }
        }
        if (resPsControlCountSub) {
            resPsControlCountSub.textContent = 'N₀ = ' + formatNumber(r.n_control) + ' control units';
        }

        if (resPsBoundaryCount) {
            const nearZ = diag.scores_near_zero || 0;
            const nearO = diag.scores_near_one || 0;
            resPsBoundaryCount.textContent = formatNumber(nearZ + nearO);
        }
        if (resPsBoundarySub) {
            const zThresh = diag.near_zero_threshold !== undefined ? diag.near_zero_threshold : 0.05;
            const oThresh = diag.near_one_threshold !== undefined ? diag.near_one_threshold : 0.95;
            resPsBoundarySub.textContent = (diag.scores_near_zero || 0) + ' (< ' + zThresh + '), ' + (diag.scores_near_one || 0) + ' (> ' + oThresh + ')';
        }

        const meta = r.model_metadata || {};
        if (resPsPseudoR2) {
            resPsPseudoR2.textContent = (meta.pseudo_r2 !== null && meta.pseudo_r2 !== undefined) ? Number(meta.pseudo_r2).toFixed(4) : 'N/A';
        }
        if (resPsConvergedSub) {
            const conv = meta.converged ? 'Converged (' + (meta.n_iterations || 0) + ' iter)' : 'Did not converge';
            resPsConvergedSub.textContent = conv;
        }

        // Render Pre-Adjustment Covariate Imbalance Table
        if (psBalanceTbody) {
            psBalanceTbody.innerHTML = '';
            const balanceMap = r.covariate_balance || r.unadjusted_covariate_balance || {};
            const keys = Object.keys(balanceMap);

            if (keys.length === 0) {
                const tr = document.createElement('tr');
                const td = document.createElement('td');
                td.colSpan = 6;
                td.textContent = 'No covariate balance statistics available.';
                td.style.textAlign = 'center';
                tr.appendChild(td);
                psBalanceTbody.appendChild(tr);
            } else {
                keys.forEach(function (colName) {
                    const row = balanceMap[colName];
                    const tr = document.createElement('tr');

                    // 1. Name
                    const tdName = document.createElement('td');
                    tdName.style.fontWeight = '600';
                    tdName.textContent = colName;
                    tr.appendChild(tdName);

                    // 2. Type
                    const tdType = document.createElement('td');
                    const spanType = document.createElement('span');
                    spanType.className = 'col-type-tag';
                    spanType.textContent = row.covariate_type || 'numeric';
                    tdType.appendChild(spanType);
                    tr.appendChild(tdType);

                    // 3. Treated Mean
                    const tdTMean = document.createElement('td');
                    tdTMean.textContent = (row.mean_treated !== null && row.mean_treated !== undefined) ? Number(row.mean_treated).toFixed(2) : '—';
                    tr.appendChild(tdTMean);

                    // 4. Control Mean
                    const tdCMean = document.createElement('td');
                    tdCMean.textContent = (row.mean_control !== null && row.mean_control !== undefined) ? Number(row.mean_control).toFixed(2) : '—';
                    tr.appendChild(tdCMean);

                    // 5. SMD
                    const tdSmd = document.createElement('td');
                    if (row.smd !== null && row.smd !== undefined) {
                        const smdVal = Number(row.smd);
                        const prefix = smdVal > 0 ? '+' : '';
                        tdSmd.textContent = prefix + smdVal.toFixed(3);
                    } else {
                        tdSmd.textContent = '—';
                    }
                    tr.appendChild(tdSmd);

                    // 6. Balance Status Badge
                    const tdStatus = document.createElement('td');
                    const spanStatus = document.createElement('span');
                    if (row.is_balanced === true) {
                        spanStatus.className = 'type-badge type-boolean';
                        spanStatus.textContent = 'Balanced (|SMD| ≤ 0.10)';
                    } else if (row.is_balanced === false) {
                        spanStatus.className = 'type-badge type-string';
                        spanStatus.textContent = 'Imbalanced (|SMD| > 0.10)';
                        spanStatus.style.backgroundColor = '#FEE2E2';
                        spanStatus.style.color = '#B91C1C';
                    } else {
                        spanStatus.className = 'type-badge type-unknown';
                        spanStatus.textContent = row.note || 'Categorical (unweighted)';
                    }
                    tdStatus.appendChild(spanStatus);
                    tr.appendChild(tdStatus);

                    psBalanceTbody.appendChild(tr);
                });
            }
        }

        if (resPsNumConf) {
            resPsNumConf.textContent = (r.numeric_confounders && r.numeric_confounders.length) ? r.numeric_confounders.join(', ') : 'None';
        }
        if (resPsCatConf) {
            resPsCatConf.textContent = (r.categorical_confounders && r.categorical_confounders.length) ? r.categorical_confounders.join(', ') : 'None';
        }
        if (resPsFeaturesList) {
            resPsFeaturesList.textContent = (r.model_features && r.model_features.length) ? r.model_features.join(', ') : 'None';
        }

        if (resPsInterpretationText) {
            resPsInterpretationText.textContent = r.interpretation || 'No interpretation provided.';
        }
        if (resPsAssumptionsText) {
            resPsAssumptionsText.textContent = r.assumptions || 'No identification assumptions documented.';
        }
    }

    // --- 9. Causal Analysis (IPTW ATE) Logic ---
    function showIptwStatus(message, type) {
        if (!iptwStatusBanner) return;
        iptwStatusBanner.className = 'status-indicator status-visible status-' + type;
        iptwStatusBanner.textContent = message;
    }

    function clearIptwStatus() {
        if (!iptwStatusBanner) return;
        iptwStatusBanner.className = 'status-indicator';
        iptwStatusBanner.textContent = '';
    }

    function setupIptwAnalysis(data) {
        if (!data || !data.columns) return;
        if (iptwUnloadedNotice) iptwUnloadedNotice.style.display = 'none';
        if (iptwFormContainer) iptwFormContainer.style.display = 'block';
        if (iptwResultsPanel) iptwResultsPanel.style.display = 'none';
        clearIptwStatus();

        if (iptwDatasetName) iptwDatasetName.textContent = data.filename + (data.is_benchmark ? ' [Benchmark]' : '');
        if (iptwDatasetId) iptwDatasetId.textContent = data.dataset_id;
        if (iptwDatasetRows) iptwDatasetRows.textContent = formatNumber(data.row_count);
        if (iptwDatasetCols) iptwDatasetCols.textContent = formatNumber(data.column_count);

        if (iptwTreatmentSelect) {
            iptwTreatmentSelect.innerHTML = '<option value="">-- Choose treatment column --</option>';
            let defaultTreatment = '';
            data.columns.forEach(function (col) {
                const optT = document.createElement('option');
                optT.value = col.name;
                optT.textContent = col.name + ' (' + col.inferred_type + ')';
                iptwTreatmentSelect.appendChild(optT);

                const nameLower = col.name.toLowerCase();
                if (!defaultTreatment && (col.inferred_type === 'boolean' || nameLower === 'treatment' || nameLower === 'mens')) {
                    defaultTreatment = col.name;
                }
            });
            if (defaultTreatment) iptwTreatmentSelect.value = defaultTreatment;
        }

        if (iptwOutcomeSelect) {
            iptwOutcomeSelect.innerHTML = '<option value="">-- Choose outcome column --</option>';
            let defaultOutcome = '';
            data.columns.forEach(function (col) {
                const optO = document.createElement('option');
                optO.value = col.name;
                optO.textContent = col.name + ' (' + col.inferred_type + ')';
                iptwOutcomeSelect.appendChild(optO);

                const nameLower = col.name.toLowerCase();
                if (!defaultOutcome && (col.inferred_type === 'continuous' || col.inferred_type === 'integer') &&
                    (nameLower === 'spend' || nameLower === 'outcome' || nameLower === 'y' || nameLower === 'conversion')) {
                    defaultOutcome = col.name;
                }
            });
            if (defaultOutcome) iptwOutcomeSelect.value = defaultOutcome;
        }

        // Populate confounder checkboxes
        if (iptwConfoundersContainer) {
            iptwConfoundersContainer.innerHTML = '';
            data.columns.forEach(function (col) {
                const label = document.createElement('label');
                label.className = 'confounder-checkbox-item';
                label.setAttribute('data-col', col.name);

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'iptw-confounder';
                checkbox.value = col.name;

                // Check default confounders for Hillstrom if not treatment or outcome
                const nameLower = col.name.toLowerCase();
                if (data.is_benchmark && (nameLower === 'recency' || nameLower === 'history')) {
                    checkbox.checked = true;
                }

                checkbox.addEventListener('change', function () {
                    validateIptwForm();
                });

                const spanName = document.createElement('span');
                spanName.textContent = col.name;

                const spanType = document.createElement('span');
                spanType.className = 'col-type-tag';
                spanType.textContent = col.inferred_type;

                label.appendChild(checkbox);
                label.appendChild(spanName);
                label.appendChild(spanType);

                iptwConfoundersContainer.appendChild(label);
            });
        }

        // Reset truncation inputs
        if (iptwTruncateCheckbox) iptwTruncateCheckbox.checked = false;
        if (iptwMaxWeightInput) {
            iptwMaxWeightInput.disabled = true;
            iptwMaxWeightInput.value = '';
        }
        if (iptwExtremeWeightThresholdInput) iptwExtremeWeightThresholdInput.value = '20.0';

        updateIptwConfoundersAvailability();
        validateIptwForm();
    }

    function updateIptwConfoundersAvailability() {
        if (!iptwConfoundersContainer) return;
        const tCol = iptwTreatmentSelect ? iptwTreatmentSelect.value : '';
        const oCol = iptwOutcomeSelect ? iptwOutcomeSelect.value : '';

        const items = iptwConfoundersContainer.querySelectorAll('.confounder-checkbox-item');
        items.forEach(function (item) {
            const col = item.getAttribute('data-col');
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (!checkbox) return;

            if (col && (col === tCol || col === oCol)) {
                checkbox.checked = false;
                checkbox.disabled = true;
                item.classList.add('disabled');
            } else {
                checkbox.disabled = false;
                item.classList.remove('disabled');
            }
        });
    }

    function getSelectedIptwConfounders() {
        if (!iptwConfoundersContainer) return [];
        const checkboxes = iptwConfoundersContainer.querySelectorAll('input[name="iptw-confounder"]:checked');
        const selected = [];
        checkboxes.forEach(function (cb) {
            if (!cb.disabled && cb.value) {
                selected.push(cb.value);
            }
        });
        return selected;
    }

    function validateIptwForm() {
        if (!currentDataset) {
            if (runIptwBtn) runIptwBtn.disabled = true;
            return false;
        }

        const tCol = iptwTreatmentSelect ? iptwTreatmentSelect.value : '';
        const oCol = iptwOutcomeSelect ? iptwOutcomeSelect.value : '';
        const selectedConfounders = getSelectedIptwConfounders();

        if (!tCol || !oCol) {
            clearIptwStatus();
            if (runIptwBtn) runIptwBtn.disabled = true;
            return false;
        }

        if (tCol === oCol) {
            showIptwStatus('Treatment and Outcome cannot be the same column ("' + tCol + '").', 'error');
            if (runIptwBtn) runIptwBtn.disabled = true;
            return false;
        }

        if (selectedConfounders.length === 0) {
            clearIptwStatus();
            if (runIptwBtn) runIptwBtn.disabled = true;
            return false;
        }

        for (let i = 0; i < selectedConfounders.length; i++) {
            const cCol = selectedConfounders[i];
            if (cCol === tCol) {
                showIptwStatus('Confounder "' + cCol + '" cannot also be the Treatment column.', 'error');
                if (runIptwBtn) runIptwBtn.disabled = true;
                return false;
            }
            if (cCol === oCol) {
                showIptwStatus('Confounder "' + cCol + '" cannot also be the Outcome column.', 'error');
                if (runIptwBtn) runIptwBtn.disabled = true;
                return false;
            }
        }

        const extThresh = iptwExtremeWeightThresholdInput ? parseFloat(iptwExtremeWeightThresholdInput.value) : 20.0;
        if (isNaN(extThresh) || extThresh <= 0) {
            showIptwStatus('Extreme weight diagnostic threshold must be a positive number (> 0).', 'error');
            if (runIptwBtn) runIptwBtn.disabled = true;
            return false;
        }

        if (iptwTruncateCheckbox && iptwTruncateCheckbox.checked) {
            const maxW = iptwMaxWeightInput ? parseFloat(iptwMaxWeightInput.value) : NaN;
            if (isNaN(maxW) || maxW <= 0) {
                showIptwStatus('Maximum weight truncation cap must be a positive number (> 0).', 'error');
                if (runIptwBtn) runIptwBtn.disabled = true;
                return false;
            }
        }

        clearIptwStatus();
        if (runIptwBtn) runIptwBtn.disabled = false;
        return true;
    }

    if (iptwTreatmentSelect) {
        iptwTreatmentSelect.addEventListener('change', function () {
            updateIptwConfoundersAvailability();
            validateIptwForm();
        });
    }

    if (iptwOutcomeSelect) {
        iptwOutcomeSelect.addEventListener('change', function () {
            updateIptwConfoundersAvailability();
            validateIptwForm();
        });
    }

    if (iptwTruncateCheckbox) {
        iptwTruncateCheckbox.addEventListener('change', function () {
            if (iptwMaxWeightInput) {
                iptwMaxWeightInput.disabled = !this.checked;
                if (this.checked && !iptwMaxWeightInput.value) {
                    iptwMaxWeightInput.value = '10.0';
                }
            }
            validateIptwForm();
        });
    }

    if (iptwMaxWeightInput) iptwMaxWeightInput.addEventListener('input', validateIptwForm);
    if (iptwExtremeWeightThresholdInput) iptwExtremeWeightThresholdInput.addEventListener('input', validateIptwForm);

    if (runIptwBtn) {
        runIptwBtn.addEventListener('click', function (e) {
            e.preventDefault();
            if (!validateIptwForm()) return;
            if (!currentDataset) {
                showIptwStatus('No active dataset found. Please load or upload a dataset first.', 'error');
                return;
            }

            const selectedConfounders = getSelectedIptwConfounders();
            const isTrunc = iptwTruncateCheckbox ? iptwTruncateCheckbox.checked : false;
            const payload = {
                treatment_col: iptwTreatmentSelect.value,
                outcome_col: iptwOutcomeSelect.value,
                confounder_cols: selectedConfounders,
                drop_missing: iptwDropMissingCheckbox ? iptwDropMissingCheckbox.checked : false,
                extreme_weight_threshold: iptwExtremeWeightThresholdInput ? parseFloat(iptwExtremeWeightThresholdInput.value) || 20.0 : 20.0,
                truncate_weights: isTrunc,
            };

            if (isTrunc && iptwMaxWeightInput && iptwMaxWeightInput.value) {
                payload.max_weight = parseFloat(iptwMaxWeightInput.value);
            }

            const treatedVal = iptwTreatedValInput ? iptwTreatedValInput.value.trim() : '';
            if (treatedVal !== '') payload.treated_value = treatedVal;

            const untreatedVal = iptwUntreatedValInput ? iptwUntreatedValInput.value.trim() : '';
            if (untreatedVal !== '') payload.untreated_value = untreatedVal;

            runIptwBtn.disabled = true;
            if (iptwResultsPanel) iptwResultsPanel.style.display = 'none';
            showIptwStatus('Fitting propensity model and calculating IPTW treatment effect... Please wait.', 'info');

            fetch('/api/dataset/' + encodeURIComponent(currentDataset.dataset_id) + '/causal/iptw-ate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(payload),
            })
                .then(function (res) {
                    if (!res.ok) {
                        return res.json().then(function (err) {
                            throw new Error(err.detail || 'IPTW request failed with HTTP ' + res.status);
                        });
                    }
                    return res.json();
                })
                .then(function (results) {
                    renderIptwResults(results);
                    showIptwStatus('IPTW analysis complete: evaluated ' + formatNumber(results.n_total) + ' observations from ' + results.dataset_filename + '.', 'success');
                })
                .catch(function (err) {
                    showIptwStatus('IPTW Analysis Error: ' + err.message, 'error');
                })
                .finally(function () {
                    runIptwBtn.disabled = false;
                });
        });
    }

    function renderIptwResults(r) {
        if (!r) return;
        if (iptwResultsPanel) iptwResultsPanel.style.display = 'block';

        if (resIptwDatasetName) resIptwDatasetName.textContent = r.dataset_filename || '—';
        if (resIptwTotalRows) resIptwTotalRows.textContent = formatNumber(r.total_dataset_rows);
        if (resIptwEvalRows) resIptwEvalRows.textContent = formatNumber(r.n_total);
        if (resIptwDroppedRows) resIptwDroppedRows.textContent = formatNumber(r.rows_dropped);

        if (resIptwAte) {
            const prefix = r.iptw_ate > 0 ? '+' : '';
            resIptwAte.textContent = prefix + Number(r.iptw_ate).toFixed(4);
        }

        if (resIptwUnadjustedAte) {
            const prefix = r.unadjusted_ate > 0 ? '+' : '';
            resIptwUnadjustedAte.textContent = prefix + Number(r.unadjusted_ate).toFixed(4);
        }

        if (resIptwMeanTreated) {
            resIptwMeanTreated.textContent = (r.weighted_mean_outcome_treated !== null && r.weighted_mean_outcome_treated !== undefined) ? Number(r.weighted_mean_outcome_treated).toFixed(4) : '—';
        }
        if (resIptwTreatedSub) {
            resIptwTreatedSub.textContent = 'N₁ = ' + formatNumber(r.n_treated) + ' treated units';
        }

        if (resIptwMeanControl) {
            resIptwMeanControl.textContent = (r.weighted_mean_outcome_control !== null && r.weighted_mean_outcome_control !== undefined) ? Number(r.weighted_mean_outcome_control).toFixed(4) : '—';
        }
        if (resIptwControlSub) {
            resIptwControlSub.textContent = 'N₀ = ' + formatNumber(r.n_control) + ' control units';
        }

        if (resIptwEssTotal) {
            resIptwEssTotal.textContent = (r.ess_total !== null && r.ess_total !== undefined) ? Number(r.ess_total).toFixed(1) : '—';
        }
        if (resIptwEssSub) {
            const tFrac = (r.ess_treated_fraction !== undefined) ? (r.ess_treated_fraction * 100).toFixed(1) + '%' : '—';
            const cFrac = (r.ess_control_fraction !== undefined) ? (r.ess_control_fraction * 100).toFixed(1) + '%' : '—';
            resIptwEssSub.textContent = 'Treated ESS: ' + Number(r.ess_treated || 0).toFixed(1) + ' (' + tFrac + '), Control: ' + Number(r.ess_control || 0).toFixed(1) + ' (' + cFrac + ')';
        }

        const ws = r.weight_summary || {};
        if (resIptwWeightSummary) {
            const minW = (ws.min_weight !== undefined) ? Number(ws.min_weight).toFixed(2) : '—';
            const maxW = (ws.max_weight !== undefined) ? Number(ws.max_weight).toFixed(2) : '—';
            const meanW = (ws.mean_weight !== undefined) ? Number(ws.mean_weight).toFixed(2) : '—';
            resIptwWeightSummary.textContent = '[' + minW + ', ' + maxW + '] (μ=' + meanW + ')';
        }
        if (resIptwWeightSub) {
            const nExt = ws.n_extreme_weights || 0;
            const thresh = ws.extreme_weight_threshold || 20.0;
            const isTrunc = ws.is_truncated ? 'Yes (capped at ' + ws.truncation_max_weight + ', ' + (ws.rows_truncated || 0) + ' clipped)' : 'No';
            resIptwWeightSub.textContent = nExt + ' extreme weights (> ' + thresh + ') | Truncated: ' + isTrunc;
        }

        // Render Covariate Balance Table Before vs After Weighting
        if (iptwBalanceTbody) {
            iptwBalanceTbody.innerHTML = '';
            const beforeMap = r.covariate_balance_before || {};
            const afterMap = r.covariate_balance_after || {};
            const allKeys = Object.keys(beforeMap);

            if (allKeys.length === 0) {
                const tr = document.createElement('tr');
                const td = document.createElement('td');
                td.colSpan = 6;
                td.textContent = 'No covariate balance statistics available.';
                td.style.textAlign = 'center';
                tr.appendChild(td);
                iptwBalanceTbody.appendChild(tr);
            } else {
                allKeys.forEach(function (colName) {
                    const rowB = beforeMap[colName] || {};
                    const rowA = afterMap[colName] || {};
                    const tr = document.createElement('tr');

                    // 1. Confounder name
                    const tdName = document.createElement('td');
                    tdName.style.fontWeight = '600';
                    tdName.textContent = colName;
                    tr.appendChild(tdName);

                    // 2. Type
                    const tdType = document.createElement('td');
                    const spanType = document.createElement('span');
                    spanType.className = 'col-type-tag';
                    spanType.textContent = rowB.covariate_type || rowA.covariate_type || 'numeric';
                    tdType.appendChild(spanType);
                    tr.appendChild(tdType);

                    // 3. Before SMD
                    const tdBef = document.createElement('td');
                    if (rowB.smd !== null && rowB.smd !== undefined) {
                        const smdB = Number(rowB.smd);
                        tdBef.textContent = (smdB > 0 ? '+' : '') + smdB.toFixed(3);
                    } else {
                        tdBef.textContent = '—';
                    }
                    tr.appendChild(tdBef);

                    // 4. After SMD
                    const tdAft = document.createElement('td');
                    if (rowA.smd !== null && rowA.smd !== undefined) {
                        const smdA = Number(rowA.smd);
                        tdAft.textContent = (smdA > 0 ? '+' : '') + smdA.toFixed(3);
                    } else {
                        tdAft.textContent = '—';
                    }
                    tr.appendChild(tdAft);

                    // 5. SMD Change (Delta)
                    const tdDelta = document.createElement('td');
                    if (rowB.smd !== null && rowB.smd !== undefined && rowA.smd !== null && rowA.smd !== undefined) {
                        const absB = Math.abs(Number(rowB.smd));
                        const absA = Math.abs(Number(rowA.smd));
                        const diff = absA - absB;
                        if (diff < -0.001) {
                            tdDelta.textContent = diff.toFixed(3) + ' (improved)';
                            tdDelta.style.color = 'var(--deep-sage)';
                            tdDelta.style.fontWeight = '600';
                        } else if (diff > 0.001) {
                            tdDelta.textContent = '+' + diff.toFixed(3) + ' (worsened)';
                            tdDelta.style.color = '#B91C1C';
                        } else {
                            tdDelta.textContent = '0.000 (unchanged)';
                            tdDelta.style.color = 'var(--text-secondary)';
                        }
                    } else {
                        tdDelta.textContent = '—';
                    }
                    tr.appendChild(tdDelta);

                    // 6. Post-Weighting Balance Status Badge
                    const tdStatus = document.createElement('td');
                    const spanStatus = document.createElement('span');
                    if (rowA.is_balanced === true) {
                        spanStatus.className = 'type-badge type-boolean';
                        spanStatus.textContent = 'Balanced (|SMD| ≤ 0.10)';
                    } else if (rowA.is_balanced === false) {
                        spanStatus.className = 'type-badge type-string';
                        spanStatus.textContent = 'Imbalanced (|SMD| > 0.10)';
                        spanStatus.style.backgroundColor = '#FEE2E2';
                        spanStatus.style.color = '#B91C1C';
                    } else {
                        spanStatus.className = 'type-badge type-unknown';
                        spanStatus.textContent = rowA.note || 'Categorical (unweighted)';
                    }
                    tdStatus.appendChild(spanStatus);
                    tr.appendChild(tdStatus);

                    iptwBalanceTbody.appendChild(tr);
                });
            }
        }

        // Render Diagnostic Warnings
        if (iptwWarningsCard && iptwWarningsList) {
            iptwWarningsList.innerHTML = '';
            if (r.warnings && r.warnings.length > 0) {
                r.warnings.forEach(function (warnMsg) {
                    const li = document.createElement('li');
                    li.textContent = warnMsg;
                    iptwWarningsList.appendChild(li);
                });
                iptwWarningsCard.style.display = 'block';
            } else {
                iptwWarningsCard.style.display = 'none';
            }
        }

        if (resIptwInterpretationText) {
            resIptwInterpretationText.textContent = r.interpretation || 'No interpretation provided.';
        }
        if (resIptwAssumptionsText) {
            resIptwAssumptionsText.textContent = r.assumptions || 'No identification assumptions documented.';
        }
    }
});
