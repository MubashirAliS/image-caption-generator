import type React from "react";
import { useState, useRef, useCallback, useEffect, type DragEvent } from "react";
import {
  Upload,
  ImageIcon,
  Loader2,
  Sparkles,
  CheckCircle2,
  X,
  Brain,
  Cpu,
  RefreshCw,
  Volume2,
  VolumeOff,
  Zap,
  Info,
  ShieldAlert,
} from "lucide-react";

/* ── Hooks ── */

const speechSupported =
  typeof window !== "undefined" && "speechSynthesis" in window;

function useSpeech() {
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const utterRef = useRef<SpeechSynthesisUtterance | null>(null);

  const stop = useCallback(() => {
    if (!speechSupported) return;
    window.speechSynthesis.cancel();
    utterRef.current = null;
    setSpeakingId(null);
  }, []);

  const speak = useCallback(
    (text: string, id: string) => {
      if (!speechSupported) return;
      stop();
      if (!text.trim()) return;
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 0.95;
      utter.pitch = 1;
      utter.onend = () => setSpeakingId(null);
      utter.onerror = () => setSpeakingId(null);
      utterRef.current = utter;
      setSpeakingId(id);
      window.speechSynthesis.speak(utter);
    },
    [stop],
  );

  const toggle = useCallback(
    (text: string, id: string) => {
      if (speakingId === id) stop();
      else speak(text, id);
    },
    [speakingId, speak, stop],
  );

  return { speakingId, speak, toggle, stop };
}

/* ── Constants & validation ── */

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";
const ASSIST_KEY = "auto-assist";

const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png"]);
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "png"]);

function getFileExtension(name: string): string {
  const dotIdx = name.lastIndexOf(".");
  return dotIdx >= 0 ? name.slice(dotIdx + 1).toLowerCase() : "";
}

function validateFile(f: File): string | null {
  const ext = getFileExtension(f.name);

  if (!ALLOWED_EXTENSIONS.has(ext)) {
    if (["mp4", "mov", "avi", "mkv", "webm", "wmv", "flv"].includes(ext))
      return "Video files are not supported. Please upload a JPG, JPEG, or PNG image.";
    if (["zip", "rar", "7z", "tar", "gz"].includes(ext))
      return "Archive files are not supported. Please upload a JPG, JPEG, or PNG image.";
    if (["svg", "svgz"].includes(ext))
      return "SVG files are not supported. Please upload a JPG, JPEG, or PNG image.";
    if (["gif", "webp", "bmp", "tiff", "tif", "ico", "heic", "heif", "avif"].includes(ext))
      return `${ext.toUpperCase()} format is not supported. Please upload a JPG, JPEG, or PNG image.`;
    return `Unsupported file format "${ext || "unknown"}". Only JPG, JPEG, and PNG images are accepted.`;
  }

  if (!ALLOWED_MIME_TYPES.has(f.type) && f.type !== "")
    return `The file "${f.name}" does not appear to be a valid image. Please upload a JPG, JPEG, or PNG file.`;

  if (f.size > 5 * 1024 * 1024)
    return "Image must be under 5 MB. Please choose a smaller file.";

  return null;
}

/* ── Types ── */

interface CaptionResult {
  success: boolean;
  greedy_caption: string;
  beam_caption: string;
}

type Status = "idle" | "uploading" | "success" | "error";

/* ── App ── */

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<CaptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [assist, setAssist] = useState(() => {
    const stored = localStorage.getItem(ASSIST_KEY);
    return stored === null ? true : stored === "true";
  });
  const [liveMessage, setLiveMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLHeadingElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  const autoSpeakTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingAutoGenerate = useRef(false);
  const { speakingId, speak, toggle: toggleSpeech, stop: stopSpeech } = useSpeech();

  const announce = useCallback((msg: string) => {
    setLiveMessage("");
    requestAnimationFrame(() => setLiveMessage(msg));
  }, []);

  const cancelAutoSpeak = useCallback(() => {
    if (autoSpeakTimer.current) {
      clearTimeout(autoSpeakTimer.current);
      autoSpeakTimer.current = null;
    }
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => setBackendOk(d.model_loaded === true))
      .catch(() => setBackendOk(false));
  }, []);

  const toggleAssist = useCallback(() => {
    setAssist((prev) => {
      const next = !prev;
      localStorage.setItem(ASSIST_KEY, String(next));
      if (!next) { stopSpeech(); cancelAutoSpeak(); }
      announce(
        next
          ? "Auto-assist enabled. Captions will be generated and spoken automatically."
          : "Auto-assist disabled.",
      );
      return next;
    });
  }, [stopSpeech, cancelAutoSpeak, announce]);

  const generateCaption = useCallback(
    async (fileToUse: File, autoSpeak: boolean) => {
      stopSpeech();
      cancelAutoSpeak();
      setStatus("uploading");
      setError(null);
      setResult(null);
      announce("Generating captions, please wait.");

      const form = new FormData();
      form.append("image", fileToUse);

      try {
        const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `Server error (${res.status})`);
        }
        const data: CaptionResult = await res.json();
        setResult(data);
        setStatus("success");
        announce("Captions generated successfully.");
        setTimeout(() => resultsRef.current?.focus(), 100);

        if (autoSpeak && speechSupported && data.beam_caption) {
          autoSpeakTimer.current = setTimeout(() => speak(data.beam_caption, "beam"), 1500);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setStatus("error");
        announce("Caption generation failed. Please try again.");
        setTimeout(() => errorRef.current?.focus(), 100);
      }
    },
    [stopSpeech, cancelAutoSpeak, announce, speak],
  );

  const handleFile = useCallback(
    (f: File) => {
      const validationError = validateFile(f);
      if (validationError) { setError(validationError); return; }
      stopSpeech();
      cancelAutoSpeak();
      if (preview) URL.revokeObjectURL(preview);
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
      setStatus("idle");

      if (assist) pendingAutoGenerate.current = true;
      else announce(`Image uploaded: ${f.name}. Ready to generate caption.`);
    },
    [preview, stopSpeech, cancelAutoSpeak, assist, announce],
  );

  useEffect(() => {
    if (pendingAutoGenerate.current && file && assist) {
      pendingAutoGenerate.current = false;
      announce(`Image uploaded: ${file.name}. Generating caption automatically.`);
      generateCaption(file, true);
    }
  }, [file, assist, generateCaption, announce]);

  const onDrop = useCallback(
    (e: DragEvent) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); },
    [handleFile],
  );
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) handleFile(f); };
  const onDropZoneKey = (e: React.KeyboardEvent) => { if ((e.key === "Enter" || e.key === " ") && !preview) { e.preventDefault(); inputRef.current?.click(); } };

  const clearImage = () => {
    stopSpeech(); cancelAutoSpeak();
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setResult(null); setError(null); setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  };

  const onManualGenerate = () => { if (file) generateCaption(file, false); };

  const hasResults = status === "success" && result;
  const showingResults = hasResults || status === "uploading";

  return (
    <div className="min-h-screen flex flex-col bg-[var(--color-background)]">
      <div aria-live="polite" aria-atomic="true" className="sr-only">{liveMessage}</div>

      {/* ── Header ── */}
      <header className="sticky top-0 z-10 bg-[var(--color-background)]/80 backdrop-blur-xl border-b border-[var(--color-border)]">
        <div className="w-full px-8 sm:px-12 h-[60px] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[var(--color-accent-dim)] flex items-center justify-center">
              <Sparkles className="w-[18px] h-[18px] text-[var(--color-accent)]" aria-hidden="true" />
            </div>
            <span className="text-[15px] font-semibold text-[var(--color-text-primary)] tracking-tight">
              Image Caption Generator
            </span>
          </div>
          <div className="flex items-center gap-3">
            {speechSupported && <AssistToggle enabled={assist} onToggle={toggleAssist} />}
            <StatusBadge ok={backendOk} />
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col">
        {!showingResults ? (
          /* ── Landing / first screen ── */
          <div className="flex-1 flex flex-col px-8 sm:px-16 lg:px-24">

            {/* ▸ Upper zone — title area */}
            <div className="flex-[1.2] flex flex-col items-center justify-end pb-12 sm:pb-16">
              <h1 className="text-[48px] sm:text-[60px] lg:text-[68px] font-semibold tracking-[-0.02em] text-[var(--color-text-primary)] leading-[1.06] text-center">
                What does your image say?
              </h1>
              <p className="text-[18px] sm:text-[20px] text-[var(--color-text-secondary)] mt-8 sm:mt-10 leading-[1.75] text-center max-w-[580px]">
                Upload an image and the AI will generate a natural language caption using visual attention.
              </p>
            </div>

            {/* ▸ Lower zone — interaction area */}
            <div className="flex-1 flex flex-col items-center pt-12 sm:pt-16">
              <div className="w-full max-w-[860px]">

                {/* Upload bar / preview */}
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  onClick={() => !preview && inputRef.current?.click()}
                  onKeyDown={onDropZoneKey}
                  role={!preview ? "button" : undefined}
                  tabIndex={!preview ? 0 : undefined}
                  aria-label={!preview ? "Upload image. Drop a file here or press Enter to browse." : undefined}
                  className={`
                    w-full rounded-2xl border transition-all duration-200 overflow-hidden
                    ${preview ? "border-[var(--color-border)]" : "cursor-pointer group"}
                    ${dragOver
                      ? "border-[var(--color-accent)] bg-[var(--color-accent-dim)] scale-[1.005]"
                      : preview
                        ? "bg-[var(--color-surface)]"
                        : "border-[var(--color-border)] hover:border-[var(--color-border-hover)] bg-[var(--color-surface)]"
                    }
                  `}
                >
                  {preview ? (
                    <div className="relative">
                      <img
                        src={preview}
                        alt="Uploaded image preview"
                        className="w-full max-h-[440px] object-contain bg-[var(--color-surface-alt)] rounded-2xl"
                      />
                      <button
                        onClick={(e) => { e.stopPropagation(); clearImage(); }}
                        aria-label="Remove image"
                        className="absolute top-5 right-5 w-11 h-11 rounded-full bg-[var(--color-surface-elevated)]/90 backdrop-blur border border-[var(--color-border)] hover:border-[var(--color-border-hover)] flex items-center justify-center transition cursor-pointer"
                      >
                        <X className="w-5 h-5 text-[var(--color-text-secondary)]" aria-hidden="true" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-5 py-6 px-7 sm:px-8">
                      <div className="w-14 h-14 rounded-2xl bg-[var(--color-accent-dim)] flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform duration-200">
                        <ImageIcon className="w-7 h-7 text-[var(--color-accent)]" aria-hidden="true" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-[17px] sm:text-[18px] text-[var(--color-text-secondary)] group-hover:text-[var(--color-text-primary)] transition-colors leading-snug">
                          Drop an image here, or <span className="text-[var(--color-accent)] font-medium">browse files</span>
                        </p>
                        <p className="text-[14px] text-[var(--color-text-muted)] mt-1.5">
                          JPG, JPEG, and PNG &middot; Max 5 MB
                        </p>
                      </div>
                      <Upload className="w-6 h-6 text-[var(--color-text-muted)] shrink-0 group-hover:text-[var(--color-accent)] transition-colors" aria-hidden="true" />
                    </div>
                  )}
                  <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" onChange={onFileChange} className="hidden" aria-label="Choose image file" />
                </div>

                {/* File info + generate */}
                {file && (
                  <div className="mt-6 flex items-center gap-4">
                    <div className="flex-1 flex items-center gap-3 text-[14px] text-[var(--color-text-muted)] min-w-0">
                      <ImageIcon className="w-4.5 h-4.5 shrink-0" aria-hidden="true" />
                      <span className="truncate">{file.name}</span>
                      <span className="text-[var(--color-border)]">&middot;</span>
                      <span className="shrink-0">{(file.size / 1024).toFixed(0)} KB</span>
                      <button onClick={() => inputRef.current?.click()} className="ml-1 text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] cursor-pointer transition-colors shrink-0">
                        <RefreshCw className="w-4 h-4" aria-hidden="true" />
                      </button>
                    </div>
                    <button
                      onClick={onManualGenerate}
                      className="shrink-0 px-8 py-3 rounded-xl font-semibold text-[15px] flex items-center gap-2.5 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white cursor-pointer active:scale-[0.98] transition-all duration-200 shadow-lg shadow-[var(--color-accent)]/20"
                    >
                      <Upload className="w-4.5 h-4.5" aria-hidden="true" /> Generate
                    </button>
                  </div>
                )}

                {/* Error */}
                {error && (
                  <div ref={errorRef} tabIndex={-1} role="alert" className="mt-6 flex items-start gap-4 p-5 rounded-xl bg-[var(--color-error-dim)] border border-[var(--color-error)]/20 outline-none animate-in">
                    <ShieldAlert className="w-5 h-5 text-[var(--color-error)] mt-0.5 shrink-0" aria-hidden="true" />
                    <p className="text-[15px] font-medium text-[var(--color-error)] leading-relaxed">{error}</p>
                  </div>
                )}
              </div>

              {/* Chips */}
              <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-5 mt-20">
                <Chip icon={<Cpu className="w-4 h-4" />} label="Greedy" />
                <Chip icon={<Brain className="w-4 h-4" />} label="Beam Search" />
                {speechSupported && <Chip icon={<Volume2 className="w-4 h-4" />} label="Audio" />}
                <Chip icon={<Zap className="w-4 h-4" />} label="Auto-assist" active={assist} />
                <Chip icon={<ImageIcon className="w-4 h-4" />} label="JPG & PNG" />
              </div>
            </div>
          </div>
        ) : (
          /* ── Results view ── */
          <div className="max-w-4xl mx-auto w-full px-6 sm:px-10 py-10">
            <div className="grid lg:grid-cols-2 gap-8 items-start">
              {/* Left: Upload */}
              <section className="flex flex-col gap-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-[12px] font-bold text-[var(--color-text-muted)] uppercase tracking-[0.14em]">
                    Upload Image
                  </h3>
                  <button
                    onClick={() => inputRef.current?.click()}
                    className="text-[12px] text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] font-medium flex items-center gap-1.5 cursor-pointer transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" aria-hidden="true" /> Change
                  </button>
                </div>

                <div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden"
                >
                  {preview && (
                    <div className="relative">
                      <img src={preview} alt="Uploaded image preview" className="w-full max-h-[420px] object-contain bg-[var(--color-surface-alt)] rounded-2xl" />
                      <button
                        onClick={clearImage}
                        aria-label="Remove image"
                        className="absolute top-3 right-3 w-9 h-9 rounded-full bg-[var(--color-surface-elevated)] border border-[var(--color-border)] hover:border-[var(--color-border-hover)] flex items-center justify-center transition cursor-pointer"
                      >
                        <X className="w-4 h-4 text-[var(--color-text-secondary)]" aria-hidden="true" />
                      </button>
                    </div>
                  )}
                  <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" onChange={onFileChange} className="hidden" aria-label="Choose image file" />
                </div>

                {file && (
                  <div className="flex items-center gap-2 text-[12px] text-[var(--color-text-muted)]">
                    <ImageIcon className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
                    <span className="truncate max-w-[240px]">{file.name}</span>
                    <span className="text-[var(--color-border)]">&middot;</span>
                    <span>{(file.size / 1024).toFixed(0)} KB</span>
                  </div>
                )}

                <button
                  onClick={onManualGenerate}
                  disabled={!file || status === "uploading"}
                  className={`
                    w-full py-3 rounded-xl font-semibold text-[14px] flex items-center justify-center gap-2.5 transition-all duration-200
                    ${!file || status === "uploading"
                      ? "bg-[var(--color-surface-alt)] text-[var(--color-text-muted)] cursor-not-allowed border border-[var(--color-border)]"
                      : "bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white cursor-pointer active:scale-[0.98]"
                    }
                  `}
                >
                  {status === "uploading" ? (
                    <><Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Generating…</>
                  ) : (
                    <><Upload className="w-4 h-4" aria-hidden="true" /> Regenerate</>
                  )}
                </button>

                {error && (
                  <div ref={errorRef} tabIndex={-1} role="alert" className="flex items-start gap-3 p-4 rounded-xl bg-[var(--color-error-dim)] border border-[var(--color-error)]/20 outline-none animate-in">
                    <ShieldAlert className="w-4 h-4 text-[var(--color-error)] mt-0.5 shrink-0" aria-hidden="true" />
                    <p className="text-[13px] font-medium text-[var(--color-error)] leading-relaxed">{error}</p>
                  </div>
                )}
              </section>

              {/* Right: Results */}
              <section className="flex flex-col gap-5">
                <h3 ref={resultsRef} tabIndex={-1} className="text-[12px] font-bold text-[var(--color-text-muted)] uppercase tracking-[0.14em] outline-none">
                  Generated Captions
                </h3>

                {status === "uploading" && (
                  <div className="flex-1 flex flex-col items-center justify-center rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-14">
                    <div className="w-14 h-14 rounded-full bg-[var(--color-accent-dim)] flex items-center justify-center mb-5">
                      <Loader2 className="w-7 h-7 text-[var(--color-accent)] animate-spin" aria-hidden="true" />
                    </div>
                    <p className="text-[16px] font-semibold text-[var(--color-text-primary)]">Analyzing image…</p>
                    <p className="text-[13px] text-[var(--color-text-muted)] mt-1.5">This may take a few seconds</p>
                  </div>
                )}

                {hasResults && (
                  <div className="flex flex-col gap-4 animate-in">
                    <div className="flex items-center gap-2 px-0.5">
                      <CheckCircle2 className="w-4 h-4 text-[var(--color-success)]" aria-hidden="true" />
                      <span className="text-[12px] font-bold text-[var(--color-success)] uppercase tracking-wide">Captions ready</span>
                    </div>

                    <CaptionCard
                      icon={<Cpu className="w-4 h-4" />}
                      label="Greedy Search"
                      tag="Fast"
                      caption={result.greedy_caption}
                      speechId="greedy"
                      speakingId={speakingId}
                      onToggleSpeech={toggleSpeech}
                      showAudio={speechSupported}
                    />
                    <CaptionCard
                      icon={<Brain className="w-4 h-4" />}
                      label="Beam Search"
                      tag="Recommended"
                      caption={result.beam_caption}
                      highlighted
                      speechId="beam"
                      speakingId={speakingId}
                      onToggleSpeech={toggleSpeech}
                      showAudio={speechSupported}
                    />
                  </div>
                )}
              </section>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-[var(--color-border)]">
        <div className="px-8 sm:px-12 py-4 flex items-center justify-center gap-2 text-[12px] text-[var(--color-text-muted)]">
          <span>Image Caption Generation</span>
          <span>&middot;</span>
          <span>InceptionV3 + Visual Attention</span>
        </div>
      </footer>
    </div>
  );
}

/* ── Sub-components ── */

function Chip({ icon, label, active }: { icon: React.ReactNode; label: string; active?: boolean }) {
  return (
    <span className={`
      inline-flex items-center gap-2.5 text-[14px] font-medium px-5 py-2.5 rounded-full border transition-colors
      ${active
        ? "bg-[var(--color-accent-dim)] text-[var(--color-accent)] border-[var(--color-accent)]/30"
        : "bg-[var(--color-surface)] text-[var(--color-text-muted)] border-[var(--color-border)]"
      }
    `}>
      {icon}
      {label}
    </span>
  );
}

function AssistToggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const openTooltip = () => { if (tooltipTimeout.current) clearTimeout(tooltipTimeout.current); setShowTooltip(true); };
  const closeTooltip = () => { tooltipTimeout.current = setTimeout(() => setShowTooltip(false), 200); };

  return (
    <div className="relative flex items-center gap-1">
      <button
        role="switch"
        aria-checked={enabled}
        aria-label="Auto-assist mode"
        onClick={onToggle}
        className={`
          flex items-center gap-2 text-[12px] font-medium px-3 py-1.5 rounded-full transition-all cursor-pointer border
          ${enabled
            ? "bg-[var(--color-accent-dim)] text-[var(--color-accent)] border-[var(--color-accent)]/30"
            : "bg-[var(--color-surface)] text-[var(--color-text-muted)] border-[var(--color-border)] hover:border-[var(--color-border-hover)]"
          }
        `}
      >
        <Zap className="w-3.5 h-3.5" aria-hidden="true" />
        Auto-assist
        <span className={`relative w-7 h-4 rounded-full transition-colors ${enabled ? "bg-[var(--color-accent)]" : "bg-[var(--color-border)]"}`}>
          <span className={`absolute top-[2px] w-3 h-3 rounded-full transition-all ${enabled ? "left-[13px] bg-white" : "left-[2px] bg-[var(--color-text-muted)]"}`} />
        </span>
      </button>

      <div className="relative" onMouseEnter={openTooltip} onMouseLeave={closeTooltip} onFocus={openTooltip} onBlur={closeTooltip}>
        <button aria-label="Learn more about Auto-assist" className="w-6 h-6 rounded-full flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-accent)] transition-colors cursor-pointer">
          <Info className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
        {showTooltip && (
          <div role="tooltip" onMouseEnter={openTooltip} onMouseLeave={closeTooltip} className="absolute right-0 top-full mt-2 w-[280px] bg-[var(--color-surface-elevated)] border border-[var(--color-border)] rounded-xl shadow-2xl p-4 z-50 animate-in">
            <div className="absolute -top-1.5 right-3 w-3 h-3 rotate-45 bg-[var(--color-surface-elevated)] border-l border-t border-[var(--color-border)]" />
            <p className="text-[12px] font-bold text-[var(--color-text-primary)] mb-2.5 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-[var(--color-accent)]" aria-hidden="true" /> Auto-assist Mode
            </p>
            <div className="space-y-2.5 text-[12px] leading-relaxed text-[var(--color-text-secondary)]">
              <div>
                <p className="font-semibold text-[var(--color-success)] mb-0.5 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" aria-hidden="true" /> Enabled
                </p>
                <p>Captions generated automatically on upload. Best caption read aloud.</p>
              </div>
              <div>
                <p className="font-semibold text-[var(--color-text-muted)] mb-0.5 flex items-center gap-1">
                  <X className="w-3 h-3" aria-hidden="true" /> Disabled
                </p>
                <p>Upload first, then click Generate manually. No auto speech.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ ok }: { ok: boolean | null }) {
  if (ok === null)
    return (
      <span className="text-[12px] text-[var(--color-text-muted)] flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[var(--color-border)]" role="status">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-warning)] pulse-glow" aria-hidden="true" />
        Connecting
      </span>
    );
  if (!ok)
    return (
      <span className="text-[12px] font-medium text-[var(--color-error)] flex items-center gap-1.5 bg-[var(--color-error-dim)] px-3 py-1.5 rounded-full" role="status">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-error)]" aria-hidden="true" />
        Offline
      </span>
    );
  return (
    <span className="text-[12px] font-medium text-[var(--color-success)] flex items-center gap-1.5 bg-[var(--color-success-dim)] px-3 py-1.5 rounded-full" role="status">
      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" aria-hidden="true" />
      Ready
    </span>
  );
}

function CaptionCard({
  icon, label, tag, caption, highlighted, speechId, speakingId, onToggleSpeech, showAudio,
}: {
  icon: React.ReactNode; label: string; tag: string; caption: string; highlighted?: boolean;
  speechId: string; speakingId: string | null; onToggleSpeech: (text: string, id: string) => void; showAudio: boolean;
}) {
  const isSpeaking = speakingId === speechId;
  return (
    <div className={`rounded-2xl border p-6 transition-all ${
      highlighted
        ? "border-[var(--color-accent)]/25 bg-[var(--color-accent-dim)]"
        : "border-[var(--color-border)] bg-[var(--color-surface)]"
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className={highlighted ? "text-[var(--color-accent)]" : "text-[var(--color-text-muted)]"}>{icon}</span>
          <span className="text-[14px] font-semibold text-[var(--color-text-primary)]">{label}</span>
        </div>
        <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-md ${
          highlighted ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-surface-alt)] text-[var(--color-text-muted)]"
        }`}>{tag}</span>
      </div>
      <p className="text-[17px] leading-relaxed font-medium capitalize text-[var(--color-text-primary)]">
        &ldquo;{caption}&rdquo;
      </p>
      {showAudio && (
        <div className="mt-4 pt-4 border-t border-[var(--color-border)]/50">
          <button
            onClick={() => onToggleSpeech(caption, speechId)}
            aria-pressed={isSpeaking}
            className={`flex items-center gap-2 text-[13px] font-medium px-4 py-2 rounded-lg transition-all cursor-pointer ${
              isSpeaking
                ? "bg-[var(--color-accent)] text-white"
                : highlighted
                  ? "bg-[var(--color-accent-dim)] text-[var(--color-accent)] hover:bg-[var(--color-accent)]/20"
                  : "bg-[var(--color-surface-alt)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]"
            }`}
          >
            {isSpeaking ? <><VolumeOff className="w-4 h-4" aria-hidden="true" /> Stop</> : <><Volume2 className="w-4 h-4" aria-hidden="true" /> Play Audio</>}
          </button>
        </div>
      )}
    </div>
  );
}
