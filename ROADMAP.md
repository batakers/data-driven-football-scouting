# Roadmap: Scouting Tool Enhancement

Dokumen ini menjadi acuan pengerjaan improvement project agar dashboard lebih mendekati tool yang bisa dipakai scout profesional.

Semua phase di bawah sudah divalidasi terhadap ketersediaan dataset. Tidak ada phase yang membutuhkan data baru dari luar.

---

## Phase 8 — League-Strength Context

**Tujuan:** Memberikan konteks kekuatan liga pada setiap player, sehingga scout bisa langsung tahu apakah performa player berasal dari liga kompetitif atau tidak.

**Data yang dipakai:**
- `data/raw/competitions.csv` → `sub_type`, `country_name`, `confederation`
- `data/raw/appearances.csv` → `competition_id` per match
- `data/raw/player_valuations.csv` → `player_club_domestic_competition_id`

**Deliverables:**

1. **League tier mapping** (`src/league_tiers.py`)
   - Tier 1: Top 5 (GB1, ES1, L1, IT1, FR1)
   - Tier 2: Strong European (NL1, PO1, BE1, TR1, A1, SC1)
   - Tier 3: Competitive (RU1, UKR1, GR1, DK1, SE1, NO1, PL1, RO1, SER1)
   - Tier 4: Non-European / Lower (SA1, BRA1, MLS1, ARG1, MEX1, JAP1, KR1, AUS1, dll)
   - Mapping bisa di-override manual jika ada liga yang naik/turun reputasi

2. **Integrasi ke pipeline**
   - Tambahkan kolom `league_tier` dan `league_tier_label` ke `featured_players.csv` dan `predictions_per_player.csv`
   - Tambahkan ke similarity engine pool

3. **Dashboard update**
   - Badge tier di samping nama liga (contoh: "Premier League · Tier 1")
   - Warning otomatis di rationale jika player dari Tier 3-4: "Performance belum teruji di liga top — verifikasi adaptability"
   - Filter by league tier di overview tab

**Validasi:**
- Semua player di shortlist harus punya `league_tier` assigned
- Coverage target: 100% player yang punya `competition_id`

**Status:** ✅ Selesai

---

## Phase 9 — Contract Awareness

**Tujuan:** Menambahkan informasi kontrak agar scout bisa melihat urgency dan feasibility transfer.

**Data yang dipakai:**
- `data/raw/player_bio.csv` → `contract_until` (137,731 rows)
- `data/raw/players.csv` → `contract_expiration_date` (31,214 rows)

**Deliverables:**

1. **Contract enrichment** (`src/enrich_contract.py`)
   - Parse `contract_until` dari player_bio (format ISO datetime)
   - Fallback ke `contract_expiration_date` dari players.csv
   - Hitung `contract_months_remaining` relatif terhadap valuation date
   - Classify: "Expiring (<12 months)", "Short (12-24 months)", "Medium (24-36 months)", "Long (>36 months)"

2. **Integrasi ke scouting layer**
   - Tambahkan `contract_status` ke shortlist dan similarity results
   - Player dengan kontrak expiring mendapat flag khusus di rationale

3. **Dashboard update**
   - Kolom "Contract Status" di overview table
   - Badge "⏰ Expiring" untuk player dengan <12 bulan kontrak
   - Filter by contract status
   - Rationale: "Kontrak tersisa X bulan — window negosiasi terbuka" atau "Kontrak panjang — transfer fee kemungkinan tinggi"

**Validasi:**
- Coverage target: >60% player di shortlist punya contract info
- Tidak ada contract_months_remaining negatif yang lolos tanpa flag "Expired/Unknown"

**Status:** ✅ Selesai

---

## Phase 10 — Personalized Scouting Rationale

**Tujuan:** Mengubah rationale dari template generik menjadi penjelasan spesifik per player yang reference data aktual.

**Data yang dipakai:**
- Semua data yang sudah ada di pipeline (profile metrics, league tier, contract, role metadata)

**Deliverables:**

1. **Rewrite `src/scouting_rationale.py`**
   - `build_rationale_summary()` → generate kalimat unik per player berdasarkan:
     - Posisi dan profile metrics terkuat (contoh: "Ball Progression di top 15% midfielder Ligue 1")
     - League tier context (contoh: "Performa dari Tier 2 liga — butuh validasi di level lebih tinggi")
     - Contract situation (contoh: "Kontrak habis Juni 2026 — window negosiasi segera terbuka")
     - Age trajectory (contoh: "21 tahun dengan 2,400 minutes — sample size kuat untuk usia ini")
     - Value gap magnitude (contoh: "Estimated value 3.2x lipat dari market price saat ini")
   - `build_scout_checks()` → generate checks kontekstual:
     - Jika Tier 3-4: "Verify performance translates to higher competition level"
     - Jika <1500 minutes: "Limited senior sample — check youth international record"
     - Jika role mismatch di similarity: "Confirm tactical deployment via video analysis"
     - Jika contract expiring: "Check if club willing to sell or player wants to leave"
     - Jika GK: "Specialist metrics unavailable — prioritize video and match report review"

2. **Rationale scoring**
   - Setiap player mendapat "confidence level" berdasarkan data completeness:
     - High: enriched stats + league tier 1-2 + contract info + >1800 mins
     - Medium: basic stats + any tier + >900 mins
     - Low: limited data points

3. **Dashboard update**
   - Rationale section menampilkan teks yang berbeda per player
   - Confidence badge (High/Medium/Low) di samping suggested action
   - Tooltip explaining kenapa confidence level tersebut

**Validasi:**
- Tidak ada dua player dengan rationale text yang identik (kecuali edge case data identical)
- Setiap rationale harus reference minimal 2 data points spesifik dari player tersebut

**Status:** ✅ Selesai

---

## Phase 11 — Recency Weighting (Form Signal)

**Tujuan:** Menambahkan sinyal tren performa terkini agar scout bisa melihat apakah player sedang naik atau turun form.

**Data yang dipakai:**
- `data/raw/appearances.csv` → `date`, `goals`, `assists`, `minutes_played`, `yellow_cards`, `red_cards`

**Deliverables:**

1. **Split-window feature engineering** (`src/feature_engineering.py` update)
   - Window A: 0-180 hari sebelum valuation (recent form)
   - Window B: 181-365 hari sebelum valuation (earlier form)
   - Hitung per-90 metrics untuk masing-masing window
   - `form_trend_goals` = goals_per_90_windowA - goals_per_90_windowB
   - `form_trend_assists` = assists_per_90_windowA - assists_per_90_windowB
   - `form_trend_minutes` = minutes_windowA / minutes_windowB (ratio)
   - Classify: "Rising Form", "Stable", "Declining Form"

2. **Integrasi ke dashboard**
   - Kolom "Form" di overview table dengan arrow indicator (↑ Rising, → Stable, ↓ Declining)
   - Rationale reference: "Performa 6 bulan terakhir menunjukkan tren naik — goals/90 meningkat 40% vs semester sebelumnya"

3. **Integrasi ke rationale (Phase 10 dependency)**
   - Rising form + undervalued = stronger signal
   - Declining form + undervalued = caution flag

**Batasan yang diterima:**
- Hanya bisa track trend di goals/assists/cards/minutes
- Tidak bisa track progressive passes, xG trend, dll (data tidak tersedia untuk semua liga)
- Player dengan <450 minutes di salah satu window → form trend "Insufficient Data"

**Validasi:**
- Form trend hanya dihitung jika kedua window punya minimal 450 minutes
- Tidak ada division by zero pada form_trend_minutes

**Status:** ✅ Selesai

---

## Phase 12 — Export & Workflow Completion

**Tujuan:** Memungkinkan scout menyelesaikan workflow mereka — dari discovery sampai report yang bisa di-share.

**Data yang dipakai:**
- Tidak butuh data tambahan. Murni fitur UI/UX.

**Deliverables:**

1. **Excel export**
   - Tombol "Download Shortlist" di overview tab → Excel file dengan semua kolom + league tier + contract + form
   - Tombol "Download Comparison" di alternatives tab → Excel file hasil similarity search
   - Include metadata sheet: tanggal export, filter yang dipakai, methodology note

2. **PDF scouting brief** (per player)
   - Tombol "Generate Scouting Brief" di rationale tab
   - Isi: Player profile, key signals, scout checks, similar alternatives (top 3), evidence table
   - Format: clean, printable, bisa di-share ke sporting director
   - Library: `fpdf2` atau `reportlab`

3. **Session watchlist**
   - Streamlit session_state based watchlist
   - Tombol "Add to Watchlist" di setiap player row
   - Tab "My Watchlist" yang menampilkan semua player yang di-bookmark dalam session
   - Export watchlist ke Excel
   - Note: tidak persistent antar session (limitasi Streamlit), tapi cukup untuk satu sesi kerja

**Validasi:**
- Excel file bisa dibuka tanpa error di Excel/Google Sheets
- PDF readable dan formatting tidak broken
- Watchlist survive page navigation dalam satu session

**Status:** ✅ Selesai

---

## Phase 13 — Deployment

**Tujuan:** Dashboard bisa diakses tanpa clone repo, sehingga siapapun (recruiter, scout, hiring manager) bisa langsung mencoba.

**Deliverables:**

1. **Streamlit Cloud deployment**
   - Setup `requirements.txt` yang production-ready (pin versions)
   - Pastikan semua path relative dan tidak hardcode Windows path
   - Data files yang dibutuhkan dashboard di-include atau di-host
   - `.streamlit/config.toml` untuk theme configuration

2. **Alternatif: HuggingFace Spaces**
   - Jika Streamlit Cloud punya size limit, gunakan HF Spaces
   - Dockerfile atau space.yaml configuration

3. **README update**
   - Live demo link di bagian atas README
   - Badge "Live Demo" yang clickable

**Validasi:**
- Dashboard accessible via public URL
- Load time < 10 detik untuk initial page
- Semua tab functional tanpa error

**Status:** ✅ Selesai

---

## Dependency Graph

```
Phase 8 (League Tier) ──────┐
                             ├──→ Phase 10 (Personalized Rationale)
Phase 9 (Contract) ─────────┘           │
                                        ├──→ Phase 12 (Export & Workflow)
Phase 11 (Recency/Form) ───────────────┘           │
                                                    ├──→ Phase 13 (Deployment)
                                                    │
```

- Phase 8 dan 9 bisa dikerjakan paralel (independen)
- Phase 10 butuh Phase 8 + 9 selesai (rationale reference league tier dan contract)
- Phase 11 bisa dikerjakan paralel dengan Phase 10
- Phase 12 butuh Phase 10 + 11 selesai (export harus include semua enrichment)
- Phase 13 paling akhir (deploy versi final)

---

## Prioritas Eksekusi

| Urutan | Phase | Effort | Impact untuk Scout |
|--------|-------|--------|-------------------|
| 1 | Phase 8 — League Tier | Low | High |
| 2 | Phase 9 — Contract | Low-Medium | High |
| 3 | Phase 10 — Personalized Rationale | Medium | Very High |
| 4 | Phase 11 — Recency/Form | Medium | Medium |
| 5 | Phase 12 — Export & Workflow | Medium | High |
| 6 | Phase 13 — Deployment | Low | Very High (visibility) |

---

## Catatan

- Setiap phase yang selesai harus di-update status-nya di dokumen ini (⬜ → ✅)
- Jika ada blocker atau perubahan scope, catat di bawah phase terkait
- Model A tidak di-retrain di roadmap ini. League tier ditambahkan sebagai context/filter, bukan sebagai feature model. Retrain adalah optional extension terpisah.
- Semua improvement backward-compatible — dashboard lama tetap bisa jalan tanpa phase baru.
