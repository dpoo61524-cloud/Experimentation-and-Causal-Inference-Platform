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

        // Initialize ATE analysis workflow for this dataset
        setupAteAnalysis(data);
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
});
