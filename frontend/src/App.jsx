import React, { useState } from 'react';
import DataUploader from './components/DataUploader';
import {
  SyntheticControlChart,
  CovariateBalanceChart,
  BayesianPosteriorChart,
  DifferenceInDifferencesChart,
  DoWhyRefutationChart,
  SequentialBoundaryChart,
  WelchTTestChart,
} from './components/Charts';

const METHODS = [
  { id: 'bayesian_ab', name: 'Bayesian A/B Testing', category: 'Bayesian' },
  { id: 'psm', name: 'Propensity Score Matching', category: 'Causal' },
  { id: 'did', name: 'Difference-in-Differences', category: 'Causal' },
  { id: 'synthetic_control', name: 'Synthetic Control', category: 'Causal' },
  { id: 'dowhy_causal', name: 'DoWhy Causal', category: 'Causal' },
  { id: 'group_sequential', name: 'Group-Sequential Testing', category: 'Frequentist' },
  { id: 'two_sample_ttest', name: "Welch's Two-Sample t-Test", category: 'Frequentist' },
];

function getExecutiveConclusion(results, selectedMethod) {
  if (!results) return null;

  // 1. Group-Sequential Testing
  if (results.all_planned_boundaries_z || results.decision) {
    const k = results.checkpoint ?? 2;
    const totalK = results.total_checkpoints ?? 4;
    const zAbs = Math.abs(results.z_statistic ?? 0).toFixed(2);
    const zCrit = (results.efficacy_boundary_z ?? 2.86).toFixed(2);
    const pVal =
      results.nominal_p_value !== undefined
        ? results.nominal_p_value < 0.001
          ? '< 0.001'
          : results.nominal_p_value.toFixed(4)
        : null;

    if (results.decision === 'stop_for_efficacy') {
      return {
        badge: 'Stop Early \u2014 Statistically Significant',
        badgeColor: 'var(--sage-light)',
        text: `At Interim Look ${k} of ${totalK}, the observed test statistic (|Z| = ${zAbs}) crossed the early stopping boundary (|Z| \u2265 ${zCrit}${pVal ? `, p = ${pVal}` : ''}). Treatment B demonstrates a statistically significant improvement over Control A. You can safely conclude the experiment early and ship the change.`,
      };
    } else if (results.decision === 'stop_for_futility') {
      return {
        badge: 'Stop Early \u2014 Futility',
        badgeColor: 'var(--accent-rose)',
        text: `At Interim Look ${k} of ${totalK}, the treatment shows no positive direction or meaningful lift (|Z| = ${zAbs}). The probability of reaching statistical significance is negligible. You can terminate the experiment early to save resources.`,
      };
    } else if (results.decision === 'inconclusive') {
      return {
        badge: 'Experiment Finished \u2014 Inconclusive',
        badgeColor: 'var(--accent-amber)',
        text: `The experiment has completed its final look (${k} of ${totalK}). The observed difference (|Z| = ${zAbs}) did not reach the significance threshold (|Z| \u2265 ${zCrit}). There is insufficient statistical evidence to declare a winner.`,
      };
    } else {
      return {
        badge: 'Continue Experiment',
        badgeColor: 'var(--accent-amber)',
        text: `At Interim Look ${k} of ${totalK}, the test statistic (|Z| = ${zAbs}) has not crossed the early efficacy threshold (|Z| \u2265 ${zCrit}). The result is currently not statistically significant; continue collecting data until the next planned checkpoint.`,
      };
    }
  }

  // 2. Welch's Two-Sample t-Test
  if (results.t_statistic !== undefined) {
    const isSig = results.statistically_significant;
    const diff = (results.difference_in_means ?? (results.mean_a - results.mean_b)).toFixed(3);
    const pVal =
      results.p_value < 0.0001 ? results.p_value.toExponential(3) : results.p_value.toFixed(4);
    const ci = results.confidence_interval_95;
    const ciStr = ci ? `[${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : '';

    if (isSig) {
      return {
        badge: 'Statistically Significant Difference',
        badgeColor: 'var(--sage-light)',
        text: `Group B differs significantly from Group A (p = ${pVal}). The estimated difference in means is ${diff}${ciStr ? ` with a 95% confidence interval of ${ciStr}` : ''}. The observed gap is statistically meaningful and unlikely to be due to chance.`,
      };
    } else {
      return {
        badge: 'No Significant Difference',
        badgeColor: 'var(--accent-amber)',
        text: `No statistically significant difference was detected between Group A and Group B (p = ${pVal}). The observed difference in means (${diff}${ciStr ? `, 95% CI: ${ciStr}` : ''}) is within expected random variation.`,
      };
    }
  }

  // 3. Bayesian A/B Testing
  if (results.comparative_metrics) {
    const probB = (results.comparative_metrics.probability_b_beats_a * 100).toFixed(1);
    const lift = results.comparative_metrics.relative_lift_mean_pct.toFixed(2);
    const loss = results.comparative_metrics.expected_loss_choosing_b.toFixed(4);
    const isWinner = results.comparative_metrics.probability_b_beats_a >= 0.95;

    return {
      badge: isWinner ? 'Variant B Strongly Favored' : 'Inconclusive / More Data Needed',
      badgeColor: isWinner ? 'var(--sage-light)' : 'var(--accent-amber)',
      text: `There is a ${probB}% posterior probability that Variant B outperforms Variant A, with an expected relative lift of ${lift}%. The expected risk (potential loss) in deploying Variant B is ${loss}.`,
    };
  }

  // 4. Propensity Score Matching (PSM)
  if (results.effect_estimates && results.effect_estimates.att !== undefined) {
    const att = results.effect_estimates.att.toFixed(3);
    const pairs = results.sample_summary?.matched_pairs?.toLocaleString() ?? 'all';
    const pVal =
      results.effect_estimates.p_value < 0.001
        ? results.effect_estimates.p_value.toExponential(2)
        : results.effect_estimates.p_value.toFixed(4);
    const isPositive = results.effect_estimates.att > 0;

    return {
      badge: isPositive ? 'Positive Treatment Effect (ATT)' : 'Negative / Neutral Treatment Effect',
      badgeColor: isPositive ? 'var(--sage-light)' : 'var(--accent-amber)',
      text: `After matching ${pairs} treated units to similar control units to eliminate confounding bias, the estimated Average Treatment Effect on the Treated (ATT) is ${att > 0 ? '+' : ''}${att} (p = ${pVal}).`,
    };
  }

  // 5. Difference-in-Differences (DiD)
  if (results.did_estimate !== undefined) {
    const did = results.did_estimate.toFixed(4);
    const isPositive = results.did_estimate > 0;
    return {
      badge: isPositive ? 'Positive Net Treatment Impact' : 'Negative Net Treatment Impact',
      badgeColor: isPositive ? 'var(--sage-light)' : 'var(--accent-rose)',
      text: `Accounting for baseline trends between groups over time, the treatment created an estimated net causal impact of ${isPositive ? '+' : ''}${did} units compared to the parallel counterfactual trend.`,
    };
  }

  // 6. DoWhy Causal
  if (results.step_3_estimate || results.causal_effect !== undefined) {
    const ate = (results.step_3_estimate?.average_treatment_effect ?? results.causal_effect ?? 0).toFixed(4);
    const pVal = (results.step_3_estimate?.p_value ?? results.p_value ?? 0).toFixed(4);
    const passed = results.robustness_summary?.all_refutations_passed;

    return {
      badge: passed ? 'Causal Effect Verified & Robust' : 'Causal Effect Flagged by Sensitivity',
      badgeColor: passed ? 'var(--sage-light)' : 'var(--accent-amber)',
      text: `The estimated Average Treatment Effect (ATE) is ${ate > 0 ? '+' : ''}${ate} (p = ${pVal}). ${passed ? 'All refutation and placebo tests passed, verifying that the effect is robust against unobserved confounding.' : 'Sensitivity checks suggest the estimate may be sensitive to unobserved confounders.'}`,
    };
  }

  // 7. Synthetic Control
  if (results.fit_metrics) {
    const effect = results.fit_metrics.average_post_treatment_effect.toFixed(4);
    const ratio = results.fit_metrics.rmspe_ratio.toFixed(2);
    const isPositive = results.fit_metrics.average_post_treatment_effect > 0;

    return {
      badge: isPositive ? 'Positive Post-Treatment Divergence' : 'Negative Post-Treatment Divergence',
      badgeColor: isPositive ? 'var(--sage-light)' : 'var(--accent-amber)',
      text: `The treated unit diverged from the weighted synthetic counterfactual by an average of ${isPositive ? '+' : ''}${effect} post-intervention, with a post-to-pre RMSPE ratio of ${ratio}.`,
    };
  }

  return {
    badge: 'Analysis Complete',
    badgeColor: 'var(--sage-light)',
    text: results.interpretation || results.conclusion || 'Analysis completed successfully.',
  };
}

export default function App() {
  const [activeDataset, setActiveDataset] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedMethod, setSelectedMethod] = useState('group_sequential');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Form states
  const [convA, setConvA] = useState(50);
  const [visA, setVisA] = useState(1000);
  const [convB, setConvB] = useState(75);
  const [visB, setVisB] = useState(1000);
  const [treatCol, setTreatCol] = useState('mens');
  const [outcomeCol, setOutcomeCol] = useState('spend');
  const [timeCol, setTimeCol] = useState('newbie');
  const [confounders, setConfounders] = useState('history, recency, newbie');
  const [sampleAStr, setSampleAStr] = useState('12.5, 14.2, 13.8, 15.1, 14.9, 13.0, 14.5, 15.0');
  const [sampleBStr, setSampleBStr] = useState('16.2, 17.5, 15.9, 18.0, 17.1, 16.8, 17.9, 18.2');
  const [parsedSampleA, setParsedSampleA] = useState([12.5, 14.2, 13.8, 15.1, 14.9, 13.0, 14.5, 15.0]);
  const [parsedSampleB, setParsedSampleB] = useState([16.2, 17.5, 15.9, 18.0, 17.1, 16.8, 17.9, 18.2]);

  const onDatasetLoaded = (d) => {
    setActiveDataset(d);
    const colNames = (d.columns || []).map((c) => (typeof c === 'string' ? c : c.name));
    setColumns(colNames);

    if (colNames.length > 0) {
      if (colNames.includes('mens')) {
        setTreatCol('mens');
      } else if (colNames.includes('treatment')) {
        setTreatCol('treatment');
      } else {
        setTreatCol(colNames[0]);
      }

      if (colNames.includes('spend')) {
        setOutcomeCol('spend');
      } else if (colNames.includes('conversion')) {
        setOutcomeCol('conversion');
      } else if (colNames.includes('outcome')) {
        setOutcomeCol('outcome');
      } else if (colNames.length > 1) {
        setOutcomeCol(colNames[1]);
      }

      if (colNames.includes('time')) {
        setTimeCol('time');
      } else if (colNames.includes('newbie')) {
        setTimeCol('newbie');
      } else if (colNames.includes('period')) {
        setTimeCol('period');
      } else {
        setTimeCol(colNames[2] || colNames[0]);
      }

      const defaultConfounders = ['history', 'recency', 'newbie'].filter((c) =>
        colNames.includes(c)
      );
      if (defaultConfounders.length > 0) {
        setConfounders(defaultConfounders.join(', '));
      } else {
        const remaining = colNames.filter((c) => c !== colNames[0] && c !== colNames[1]);
        setConfounders(remaining.slice(0, 3).join(', '));
      }
    }
  };

  const handleRunAnalysis = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      let endpoint = '';
      let payload = {};

      if (selectedMethod === 'bayesian_ab') {
        endpoint = '/api/v1/bayesian-ab/pymc';
        payload = {
          conversions_a: Number(convA),
          visitors_a: Number(visA),
          conversions_b: Number(convB),
          visitors_b: Number(visB),
          draws: 2000,
          tune: 500,
        };
      } else if (selectedMethod === 'psm') {
        if (!activeDataset) {
          throw new Error('Please upload or load a dataset first.');
        }
        endpoint = `/api/v1/causal/dataset-psm/${activeDataset.dataset_id}`;
        payload = {
          treatment_col: treatCol,
          outcome_col: outcomeCol,
          confounder_cols: confounders
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean),
          caliper: 0.2,
        };
      } else if (selectedMethod === 'did') {
        if (activeDataset) {
          endpoint = `/api/dataset/${activeDataset.dataset_id}/causal/difference-in-differences`;
          payload = {
            treatment_col: treatCol,
            outcome_col: outcomeCol,
            time_col: timeCol,
          };
        } else {
          endpoint = '/api/causal/difference-in-differences';
          payload = {
            data: [
              { treatment: 1, time: 0, outcome: 10.2 },
              { treatment: 1, time: 1, outcome: 15.6 },
              { treatment: 0, time: 0, outcome: 9.8 },
              { treatment: 0, time: 1, outcome: 11.1 },
            ],
            treatment_col: 'treatment',
            time_col: 'time',
            outcome_col: 'outcome',
          };
        }
      } else if (selectedMethod === 'synthetic_control') {
        endpoint = '/api/v1/causal/synthetic-control';
        const times = [1, 2, 3, 4, 5, 6];
        const dataRows = [];
        times.forEach((t) => {
          const boost = t >= 5 ? 12.0 : 0.0;
          dataRows.push({ unit: 'Target_Unit', time: t, y: 100 + 4 * t + boost });
          dataRows.push({ unit: 'Donor_1', time: t, y: 90 + 3.5 * t });
          dataRows.push({ unit: 'Donor_2', time: t, y: 110 + 4.5 * t });
          dataRows.push({ unit: 'Donor_3', time: t, y: 85 + 3.8 * t });
        });
        payload = {
          data: dataRows,
          unit_col: 'unit',
          time_col: 'time',
          outcome_col: 'y',
          treated_unit: 'Target_Unit',
          treatment_time: 5,
        };
      } else if (selectedMethod === 'dowhy_causal' || selectedMethod === 'dowhy_pipeline') {
        if (activeDataset) {
          endpoint = `/api/v1/causal/dataset-dowhy/${activeDataset.dataset_id}`;
          payload = {
            treatment_col: treatCol,
            outcome_col: outcomeCol,
            common_causes: confounders
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean),
            run_refutations: true,
          };
        } else {
          endpoint = '/api/v1/causal/dowhy';
          const synthData = [];
          for (let i = 0; i < 200; i++) {
            const c = Math.random() * 2 - 1;
            const t = Math.random() > 0.5 ? 1 : 0;
            const y = 50 + 5.0 * t + 3.0 * c + (Math.random() - 0.5) * 2;
            synthData.push({ treat: t, outcome: y, confounder: c });
          }
          payload = {
            data: synthData,
            treatment_col: 'treat',
            outcome_col: 'outcome',
            common_causes: ['confounder'],
            run_refutations: true,
          };
        }
      } else if (selectedMethod === 'group_sequential') {
        endpoint = '/api/experiment/sequential';
        const vA = Math.max(1, Number(visA) || 1000);
        const vB = Math.max(1, Number(visB) || 1000);
        const cA = Math.min(vA, Math.max(0, Number(convA) || 0));
        const cB = Math.min(vB, Math.max(0, Number(convB) || 0));
        const plannedMaxV = Math.max(vA, vB, Math.round(((vA + vB) / 2) * (4 / 2)));
        payload = {
          checkpoint: 2,
          total_checkpoints: 4,
          conversions_a: cA,
          visitors_a: vA,
          conversions_b: cB,
          visitors_b: vB,
          max_visitors_per_group: plannedMaxV,
          alpha: 0.05,
          schedule_tolerance: null,
        };
      } else if (selectedMethod === 'two_sample_ttest') {
        endpoint = '/api/experiment/ttest';
        const arrA = sampleAStr
          .split(',')
          .map((s) => Number(s.trim()))
          .filter((n) => !isNaN(n));
        const arrB = sampleBStr
          .split(',')
          .map((s) => Number(s.trim()))
          .filter((n) => !isNaN(n));
        if (arrA.length < 2 || arrB.length < 2) {
          throw new Error('Both Sample A and Sample B must contain at least 2 numerical observations.');
        }
        setParsedSampleA(arrA);
        setParsedSampleB(arrB);
        payload = {
          group_a: arrA,
          group_b: arrB,
        };
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Analysis execution failed (HTTP ${res.status})`);
      }

      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <h1>Experimentation &amp; Causal Inference Platform</h1>
        </div>
      </header>

      <div className="grid-layout">
        {/* Left Column: Config */}
        <div>
          <DataUploader
            activeDataset={activeDataset}
            onDatasetLoaded={onDatasetLoaded}
          />

          <div className="card">
            <div className="card-title">Method</div>

            <div className="form-group">
              <select
                className="form-select"
                value={selectedMethod}
                onChange={(e) => {
                  setSelectedMethod(e.target.value);
                  setResults(null);
                }}
              >
                {METHODS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Inputs based on method */}
            {selectedMethod === 'bayesian_ab' || selectedMethod === 'group_sequential' ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div className="form-group">
                    <label className="form-label">Conversions A</label>
                    <input
                      type="number"
                      className="form-input"
                      value={convA}
                      onChange={(e) => setConvA(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Visitors A</label>
                    <input
                      type="number"
                      className="form-input"
                      value={visA}
                      onChange={(e) => setVisA(e.target.value)}
                    />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div className="form-group">
                    <label className="form-label">Conversions B</label>
                    <input
                      type="number"
                      className="form-input"
                      value={convB}
                      onChange={(e) => setConvB(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Visitors B</label>
                    <input
                      type="number"
                      className="form-input"
                      value={visB}
                      onChange={(e) => setVisB(e.target.value)}
                    />
                  </div>
                </div>
              </>
            ) : selectedMethod === 'psm' ||
              selectedMethod === 'did' ||
              selectedMethod === 'dowhy_causal' ||
              selectedMethod === 'dowhy_pipeline' ? (
              <>
                <div className="form-group">
                  <label className="form-label">Treatment Column</label>
                  {columns.length > 0 ? (
                    <select
                      className="form-select"
                      value={treatCol}
                      onChange={(e) => setTreatCol(e.target.value)}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      className="form-input"
                      value={treatCol}
                      onChange={(e) => setTreatCol(e.target.value)}
                    />
                  )}
                </div>

                <div className="form-group">
                  <label className="form-label">Outcome Column</label>
                  {columns.length > 0 ? (
                    <select
                      className="form-select"
                      value={outcomeCol}
                      onChange={(e) => setOutcomeCol(e.target.value)}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      className="form-input"
                      value={outcomeCol}
                      onChange={(e) => setOutcomeCol(e.target.value)}
                    />
                  )}
                </div>

                {selectedMethod === 'did' && (
                  <div className="form-group">
                    <label className="form-label">Time Period Column</label>
                    {columns.length > 0 ? (
                      <select
                        className="form-select"
                        value={timeCol}
                        onChange={(e) => setTimeCol(e.target.value)}
                      >
                        {columns.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        className="form-input"
                        value={timeCol}
                        onChange={(e) => setTimeCol(e.target.value)}
                      />
                    )}
                  </div>
                )}

                {selectedMethod !== 'did' && (
                  <div className="form-group">
                    <label className="form-label">Confounders (Comma separated)</label>
                    <input
                      className="form-input"
                      value={confounders}
                      onChange={(e) => setConfounders(e.target.value)}
                    />
                  </div>
                )}
              </>
            ) : selectedMethod === 'two_sample_ttest' ? (
              <>
                <div className="form-group">
                  <label className="form-label">Sample A (Comma-separated numbers)</label>
                  <input
                    className="form-input"
                    value={sampleAStr}
                    onChange={(e) => setSampleAStr(e.target.value)}
                    placeholder="e.g. 12.5, 14.2, 13.8, 15.1, 14.9, 13.0, 14.5, 15.0"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Sample B (Comma-separated numbers)</label>
                  <input
                    className="form-input"
                    value={sampleBStr}
                    onChange={(e) => setSampleBStr(e.target.value)}
                    placeholder="e.g. 16.2, 17.5, 15.9, 18.0, 17.1, 16.8, 17.9, 18.2"
                  />
                </div>
              </>
            ) : null}

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: '1rem' }}
              onClick={handleRunAnalysis}
              disabled={loading}
            >
              {loading ? 'Running...' : 'Run Analysis'}
            </button>

            {error && (
              <div style={{ color: 'var(--accent-rose)', fontSize: '0.85rem', marginTop: '0.75rem' }}>
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Results & Charts */}
        <div>
          {results ? (
            <div>
              {/* Conclusion Box */}
              {(() => {
                const conclusion = getExecutiveConclusion(results, selectedMethod);
                return (
                  <div className="conclusion-box">
                    <div className="conclusion-title">
                      <span
                        style={{
                          display: 'inline-block',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '4px',
                          fontSize: '0.78rem',
                          fontWeight: 600,
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                          color: conclusion.badgeColor,
                          border: `1px solid ${conclusion.badgeColor}`,
                        }}
                      >
                        {conclusion.badge}
                      </span>
                      <span>Conclusion</span>
                    </div>
                    <div className="conclusion-text">
                      {conclusion.text}
                    </div>
                  </div>
                );
              })()}

              <div className="stats-grid">
                {/* Group-Sequential stats */}
                {results.z_statistic !== undefined && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">Decision</div>
                      <div className="stat-value" style={{ fontSize: '0.95rem' }}>
                        {results.decision}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Z-Statistic</div>
                      <div className="stat-value">
                        {results.z_statistic > 0 ? '+' : ''}
                        {results.z_statistic?.toFixed(3)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Boundary |Z|</div>
                      <div className="stat-value">
                        {results.efficacy_boundary_z?.toFixed(3)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Nominal P-Value</div>
                      <div className="stat-value">
                        {results.nominal_p_value?.toFixed(4)}
                      </div>
                    </div>
                  </>
                )}

                {/* DiD stats */}
                {results.did_estimate !== undefined && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">DiD Estimate</div>
                      <div className="stat-value">
                        {results.did_estimate > 0 ? '+' : ''}
                        {results.did_estimate.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">&Delta; Treatment</div>
                      <div className="stat-value">
                        {(results.treatment_change ?? results.delta_treatment) > 0 ? '+' : ''}
                        {(results.treatment_change ?? results.delta_treatment)?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">&Delta; Control</div>
                      <div className="stat-value">
                        {(results.control_change ?? results.delta_control) > 0 ? '+' : ''}
                        {(results.control_change ?? results.delta_control)?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Sample Size</div>
                      <div className="stat-value">
                        {results.n_total?.toLocaleString()}
                      </div>
                    </div>
                  </>
                )}

                {/* DoWhy Causal stats */}
                {(results.step_3_estimate || results.causal_effect !== undefined) && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">Causal Effect (ATE)</div>
                      <div className="stat-value">
                        {(results.step_3_estimate?.average_treatment_effect ?? results.causal_effect) > 0 ? '+' : ''}
                        {(results.step_3_estimate?.average_treatment_effect ?? results.causal_effect)?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">P-Value</div>
                      <div className="stat-value">
                        {(results.step_3_estimate?.p_value ?? results.p_value)?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Standard Error</div>
                      <div className="stat-value">
                        {(results.step_3_estimate?.standard_error ?? results.standard_error)?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Robustness</div>
                      <div className="stat-value">
                        {results.robustness_summary?.all_refutations_passed ? 'All Passed' : 'Sensitivity Flagged'}
                      </div>
                    </div>
                  </>
                )}

                {/* Bayesian A/B stats */}
                {results.comparative_metrics && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">{'P(B > A)'}</div>
                      <div className="stat-value">
                        {(results.comparative_metrics.probability_b_beats_a * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Lift</div>
                      <div className="stat-value">
                        {results.comparative_metrics.relative_lift_mean_pct.toFixed(2)}%
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Expected Loss</div>
                      <div className="stat-value">
                        {results.comparative_metrics.expected_loss_choosing_b.toFixed(4)}
                      </div>
                    </div>
                  </>
                )}

                {/* PSM stats */}
                {results.effect_estimates && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">ATT</div>
                      <div className="stat-value">
                        {results.effect_estimates.att > 0 ? '+' : ''}
                        {results.effect_estimates.att.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">P-Value</div>
                      <div className="stat-value">
                        {results.effect_estimates.p_value.toExponential(3)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Matched Pairs</div>
                      <div className="stat-value">
                        {results.sample_summary?.matched_pairs?.toLocaleString()}
                      </div>
                    </div>
                  </>
                )}

                {/* Welch t-test stats */}
                {results.t_statistic !== undefined && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">t-Statistic</div>
                      <div className="stat-value">
                        {results.t_statistic > 0 ? '+' : ''}
                        {results.t_statistic?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">P-Value</div>
                      <div className="stat-value">{results.p_value?.toFixed(4)}</div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Difference in Means</div>
                      <div className="stat-value">
                        {results.difference_in_means?.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Degrees of Freedom</div>
                      <div className="stat-value">
                        {results.degrees_of_freedom?.toFixed(1)}
                      </div>
                    </div>
                  </>
                )}

                {/* Synthetic Control stats */}
                {results.fit_metrics && (
                  <>
                    <div className="stat-pill">
                      <div className="stat-label">Post Effect</div>
                      <div className="stat-value">
                        {results.fit_metrics.average_post_treatment_effect.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">Pre RMSPE</div>
                      <div className="stat-value">
                        {results.fit_metrics.pre_treatment_rmspe.toFixed(4)}
                      </div>
                    </div>
                    <div className="stat-pill">
                      <div className="stat-label">RMSPE Ratio</div>
                      <div className="stat-value">
                        {results.fit_metrics.rmspe_ratio.toFixed(2)}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Charts based on output */}
              {results.all_planned_boundaries_z && (
                <SequentialBoundaryChart results={results} />
              )}

              {results.step_4_refute && (
                <DoWhyRefutationChart results={results} />
              )}

              {results.treatment_mean_pre !== undefined && (
                <DifferenceInDifferencesChart results={results} />
              )}

              {results.trajectory && (
                <SyntheticControlChart trajectory={results.trajectory} />
              )}

              {results.covariate_balance && (
                <CovariateBalanceChart
                  balanceBefore={results.covariate_balance.before_matching}
                  balanceAfter={results.covariate_balance.after_matching}
                />
              )}

              {results.posteriors && (
                <BayesianPosteriorChart
                  groupA={results.posteriors.group_a}
                  groupB={results.posteriors.group_b}
                />
              )}

              {results.t_statistic !== undefined && (
                <WelchTTestChart
                  results={results}
                  sampleA={parsedSampleA}
                  sampleB={parsedSampleB}
                />
              )}
            </div>
          ) : (
            <div
              className="card"
              style={{
                textAlign: 'center',
                padding: '3rem 1.5rem',
                color: 'var(--text-muted)',
              }}
            >
              Select a method and click Run Analysis.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
