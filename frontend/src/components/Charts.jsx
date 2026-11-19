import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export function SyntheticControlChart({ trajectory }) {
  if (!trajectory || trajectory.length === 0) return null;

  const labels = trajectory.map((t) => t.time);
  const actualData = trajectory.map((t) => t.actual);
  const syntheticData = trajectory.map((t) => t.synthetic);

  const data = {
    labels,
    datasets: [
      {
        label: 'Treated Unit (Actual)',
        data: actualData,
        borderColor: '#729b79',
        backgroundColor: 'rgba(114, 155, 121, 0.2)',
        borderWidth: 2,
        tension: 0.1,
      },
      {
        label: 'Synthetic Counterfactual',
        data: syntheticData,
        borderColor: '#e5a95b',
        borderDash: [5, 5],
        borderWidth: 2,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#e6eded' } },
      title: { display: true, text: 'Actual vs. Synthetic Counterfactual Trajectory', color: '#e6eded' },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
    },
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

export function CovariateBalanceChart({ balanceBefore, balanceAfter }) {
  if (!balanceBefore || !balanceAfter) return null;

  const covariates = Object.keys(balanceBefore);
  const beforeSMDs = covariates.map((c) => balanceBefore[c].smd || balanceBefore[c].abs_smd || 0);
  const afterSMDs = covariates.map((c) => balanceAfter[c].smd || balanceAfter[c].abs_smd || 0);

  const data = {
    labels: covariates,
    datasets: [
      {
        label: 'Before Matching (Unadjusted SMD)',
        data: beforeSMDs,
        backgroundColor: 'rgba(229, 107, 107, 0.7)',
      },
      {
        label: 'After Matching (Adjusted SMD)',
        data: afterSMDs,
        backgroundColor: 'rgba(114, 155, 121, 0.7)',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#e6eded' } },
      title: { display: true, text: 'Standardized Mean Difference (Covariate Balance)', color: '#e6eded' },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
    },
  };

  return (
    <div className="chart-container">
      <Bar data={data} options={options} />
    </div>
  );
}

export function BayesianPosteriorChart({ groupA, groupB }) {
  if (!groupA || !groupB) return null;

  const meanA = groupA.mean || 0.05;
  const meanB = groupB.mean || 0.07;
  const stdA = (groupA.credible_interval[1] - groupA.credible_interval[0]) / 3.92;
  const stdB = (groupB.credible_interval[1] - groupB.credible_interval[0]) / 3.92;

  const minX = Math.max(0, Math.min(meanA - 3 * stdA, meanB - 3 * stdB));
  const maxX = Math.max(meanA + 3 * stdA, meanB + 3 * stdB);
  const step = (maxX - minX) / 40;

  const pointsX = [];
  for (let x = minX; x <= maxX; x += step) pointsX.push(x);

  const normalPDF = (x, mean, std) =>
    (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));

  const dataA = pointsX.map((x) => normalPDF(x, meanA, stdA));
  const dataB = pointsX.map((x) => normalPDF(x, meanB, stdB));

  const data = {
    labels: pointsX.map((x) => (x * 100).toFixed(2) + '%'),
    datasets: [
      {
        label: 'Group A Posterior Density',
        data: dataA,
        borderColor: '#5b92e5',
        backgroundColor: 'rgba(91, 146, 229, 0.15)',
        fill: true,
        tension: 0.3,
      },
      {
        label: 'Group B Posterior Density',
        data: dataB,
        borderColor: '#729b79',
        backgroundColor: 'rgba(114, 155, 121, 0.15)',
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#e6eded' } },
      title: { display: true, text: 'Posterior Conversion Rate Distributions', color: '#e6eded' },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7', maxTicksLimit: 8 } },
      y: { display: false },
    },
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

export function DifferenceInDifferencesChart({ results }) {
  if (!results || results.treatment_mean_pre === undefined) return null;

  const tPre = results.treatment_mean_pre;
  const tPost = results.treatment_mean_post;
  const cPre = results.control_mean_pre;
  const cPost = results.control_mean_post;
  const cChange = results.control_change ?? (cPost - cPre);
  const counterfactualPost = tPre + cChange;

  const data = {
    labels: ['Pre-Period', 'Post-Period'],
    datasets: [
      {
        label: 'Treated Group',
        data: [tPre, tPost],
        borderColor: '#729b79',
        backgroundColor: '#729b79',
        borderWidth: 3,
        pointRadius: 6,
      },
      {
        label: 'Control Group',
        data: [cPre, cPost],
        borderColor: '#5b92e5',
        backgroundColor: '#5b92e5',
        borderWidth: 3,
        pointRadius: 6,
      },
      {
        label: 'Counterfactual Parallel Trend',
        data: [tPre, counterfactualPost],
        borderColor: '#e5a95b',
        borderDash: [6, 6],
        borderWidth: 2,
        pointRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#e6eded' } },
      title: { display: true, text: 'Difference-in-Differences Parallel Trends', color: '#e6eded' },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
    },
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

export function DoWhyRefutationChart({ results }) {
  if (!results || !results.step_4_refute) return null;

  const refutes = results.step_4_refute;
  const originalAte = results.step_3_estimate?.average_treatment_effect ?? results.causal_effect ?? 0;

  const labels = ['Original ATE'];
  const values = [originalAte];
  const colors = ['#729b79'];

  Object.entries(refutes).forEach(([name, ref]) => {
    labels.push(name.replace(/_/g, ' '));
    values.push(ref.new_effect ?? ref.refuted_effect ?? 0);
    colors.push(ref.passed ? '#5b92e5' : '#e56b6b');
  });

  const data = {
    labels,
    datasets: [
      {
        label: 'Estimated Effect',
        data: values,
        backgroundColor: colors,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'DoWhy Sensitivity & Refutation Checks', color: '#e6eded' },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
    },
  };

  return (
    <div className="chart-container">
      <Bar data={data} options={options} />
    </div>
  );
}

export function SequentialBoundaryChart({ results }) {
  if (!results || !results.all_planned_boundaries_z) return null;

  const totalK = results.total_checkpoints || results.all_planned_boundaries_z.length;
  const labels = Array.from({ length: totalK }, (_, i) => `Look ${i + 1}`);
  const boundaries = results.all_planned_boundaries_z;
  const currentK = results.checkpoint;
  const currentZ = Math.abs(results.z_statistic || 0);

  const observedSeries = Array(totalK).fill(null);
  if (currentK >= 1 && currentK <= totalK) {
    observedSeries[currentK - 1] = currentZ;
  }

  const data = {
    labels,
    datasets: [
      {
        label: "O'Brien-Fleming Boundary |Z|",
        data: boundaries,
        borderColor: '#e56b6b',
        backgroundColor: 'rgba(229, 107, 107, 0.1)',
        borderWidth: 2,
        tension: 0.2,
      },
      {
        label: 'Observed |Z| at Current Look',
        data: observedSeries,
        borderColor: '#729b79',
        backgroundColor: '#729b79',
        pointRadius: 8,
        pointHoverRadius: 10,
        showLine: false,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#e6eded' } },
      title: {
        display: true,
        text: `Group-Sequential Stopping Boundary (Look ${currentK} of ${totalK})`,
        color: '#e6eded',
      },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' }, title: { display: true, text: '|Z| Score', color: '#97a6a7' } },
    },
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

export function WelchTTestChart({ results, sampleA, sampleB }) {
  if (!results || results.t_statistic === undefined) return null;

  const meanA = Number(results.mean_a ?? 0);
  const meanB = Number(results.mean_b ?? 0);
  const diff = Number(results.difference_in_means ?? (meanA - meanB));
  const ciLow = results.confidence_interval_95 ? Number(results.confidence_interval_95[0]) : null;
  const ciHigh = results.confidence_interval_95 ? Number(results.confidence_interval_95[1]) : null;

  const barData = {
    labels: ['Group A Mean', 'Group B Mean', 'Mean Difference (A - B)'],
    datasets: [
      {
        label: 'Value',
        data: [meanA, meanB, diff],
        backgroundColor: [
          'rgba(91, 146, 229, 0.75)',
          'rgba(114, 155, 121, 0.75)',
          diff >= 0 ? 'rgba(229, 169, 91, 0.75)' : 'rgba(229, 107, 107, 0.75)',
        ],
        borderColor: [
          '#5b92e5',
          '#729b79',
          diff >= 0 ? '#e5a95b' : '#e56b6b',
        ],
        borderWidth: 1.5,
      },
    ],
  };

  const pFormatted =
    results.p_value < 0.0001
      ? results.p_value.toExponential(3)
      : results.p_value.toFixed(4);

  const barOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: `Welch Two-Sample t-Test: Sample Means & Difference (t = ${results.t_statistic.toFixed(3)}, p = ${pFormatted})`,
        color: '#e6eded',
      },
      tooltip: {
        callbacks: {
          afterBody: (context) => {
            if (context[0]?.dataIndex === 2 && ciLow !== null && ciHigh !== null) {
              return `95% CI: [${ciLow.toFixed(3)}, ${ciHigh.toFixed(3)}]`;
            }
            return '';
          },
        },
      },
    },
    scales: {
      x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
      y: { grid: { color: '#344041' }, ticks: { color: '#97a6a7' } },
    },
  };

  // Compute distributions if sample data or estimates are available
  let densitySection = null;
  let stdA = 1.0;
  let stdB = 1.0;

  if (Array.isArray(sampleA) && sampleA.length > 1) {
    const varA = sampleA.reduce((acc, val) => acc + Math.pow(val - meanA, 2), 0) / (sampleA.length - 1);
    stdA = Math.sqrt(Math.max(varA, 0.0001));
  } else if (ciLow !== null && ciHigh !== null) {
    stdA = Math.max((ciHigh - ciLow) / 3.92, 0.0001);
  }

  if (Array.isArray(sampleB) && sampleB.length > 1) {
    const varB = sampleB.reduce((acc, val) => acc + Math.pow(val - meanB, 2), 0) / (sampleB.length - 1);
    stdB = Math.sqrt(Math.max(varB, 0.0001));
  } else if (ciLow !== null && ciHigh !== null) {
    stdB = Math.max((ciHigh - ciLow) / 3.92, 0.0001);
  }

  const minX = Math.min(meanA - 3.2 * stdA, meanB - 3.2 * stdB);
  const maxX = Math.max(meanA + 3.2 * stdA, meanB + 3.2 * stdB);
  if (maxX > minX) {
    const step = (maxX - minX) / 50;
    const pointsX = [];
    for (let x = minX; x <= maxX; x += step) pointsX.push(x);

    const normalPDF = (x, m, s) =>
      (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - m) / s, 2));

    const densityDataA = pointsX.map((x) => normalPDF(x, meanA, stdA));
    const densityDataB = pointsX.map((x) => normalPDF(x, meanB, stdB));

    const densityChartData = {
      labels: pointsX.map((x) => x.toFixed(2)),
      datasets: [
        {
          label: `Group A Distribution (N=${results.sample_size_a ?? '?'})`,
          data: densityDataA,
          borderColor: '#5b92e5',
          backgroundColor: 'rgba(91, 146, 229, 0.2)',
          fill: true,
          tension: 0.3,
        },
        {
          label: `Group B Distribution (N=${results.sample_size_b ?? '?'})`,
          data: densityDataB,
          borderColor: '#729b79',
          backgroundColor: 'rgba(114, 155, 121, 0.2)',
          fill: true,
          tension: 0.3,
        },
      ],
    };

    const densityOptions = {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#e6eded' } },
        title: {
          display: true,
          text: 'Estimated Group Distributions (Separation & Spread)',
          color: '#e6eded',
        },
      },
      scales: {
        x: { grid: { color: '#344041' }, ticks: { color: '#97a6a7', maxTicksLimit: 10 } },
        y: { display: false },
      },
    };

    densitySection = (
      <div className="chart-container" style={{ marginTop: '1.25rem' }}>
        <Line data={densityChartData} options={densityOptions} />
      </div>
    );
  }

  return (
    <div>
      <div className="chart-container">
        <Bar data={barData} options={barOptions} />
        {ciLow !== null && ciHigh !== null && (
          <div style={{ textAlign: 'center', marginTop: '0.65rem', fontSize: '0.85rem', color: '#97a6a7' }}>
            95% CI for Difference: <span style={{ color: '#e6eded', fontWeight: 500 }}>[{ciLow.toFixed(3)}, {ciHigh.toFixed(3)}]</span>
            {' | '}
            Degrees of Freedom: <span style={{ color: '#e6eded', fontWeight: 500 }}>{results.degrees_of_freedom?.toFixed(1)}</span>
            {' | '}
            Status: <span style={{ color: results.statistically_significant ? '#729b79' : '#e5a95b', fontWeight: 600 }}>
              {results.statistically_significant ? 'Statistically Significant (p < \u03B1)' : 'Not Significant'}
            </span>
          </div>
        )}
      </div>
      {densitySection}
    </div>
  );
}
