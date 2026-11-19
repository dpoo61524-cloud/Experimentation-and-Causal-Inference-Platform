import React, { useState } from 'react';

export default function DataUploader({ activeDataset, onDatasetLoaded }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      let res = await fetch('/api/dataset/upload', {
        method: 'POST',
        body: formData,
      });

      // Fallback to streaming text/csv if multipart fails
      if (!res.ok && res.status !== 404) {
        res = await fetch('/api/dataset/upload', {
          method: 'POST',
          headers: {
            'Content-Type': 'text/csv',
            'X-Filename': encodeURIComponent(file.name),
          },
          body: file,
        });
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed (HTTP ${res.status})`);
      }

      const data = await res.json();
      onDatasetLoaded(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>Dataset</span>
      </div>

      <div className="form-group">
        <label className="form-label">Upload CSV</label>
        <input
          type="file"
          accept=".csv"
          className="form-input"
          onChange={handleFileUpload}
          disabled={loading}
        />
      </div>

      {loading && (
        <div style={{ fontSize: '0.82rem', color: 'var(--sage-light)', marginTop: '0.5rem' }}>
          Processing...
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--accent-rose)', fontSize: '0.85rem', marginTop: '0.75rem' }}>
          {error}
        </div>
      )}

      {activeDataset && (
        <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--sage-light)', fontWeight: 600 }}>
            {activeDataset.filename || activeDataset.dataset_filename}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {(activeDataset.row_count || activeDataset.total_dataset_rows)?.toLocaleString()} rows &bull;{' '}
            {activeDataset.column_count || activeDataset.columns?.length || 0} columns
          </div>
        </div>
      )}
    </div>
  );
}
