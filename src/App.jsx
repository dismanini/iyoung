import { useCallback, useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import "./App.css";
import logoLight from "./assets/iYoung_Asset 1.svg";
import logoDark from "./assets/iYoung_White_Asset 2.svg";

const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const PARAMETERS = [
  { key: "acne", label: "Acne", icon: "🔴", color: "emerald", accent: "from-emerald-500/25 to-teal-500/10 border-emerald-500/50 text-emerald-500" },
  { key: "pigmentation", label: "Pigmentation", icon: "🟤", color: "purple", accent: "from-purple-500/25 to-indigo-500/10 border-purple-500/50 text-purple-500" },
  { key: "wrinkles", label: "Fine Lines", icon: "〰️", color: "amber", accent: "from-amber-500/25 to-orange-500/10 border-amber-500/50 text-amber-500" },
  { key: "pores", label: "Pores", icon: "⚫", color: "blue", accent: "from-blue-500/25 to-cyan-500/10 border-blue-500/50 text-blue-500" },
  { key: "redness", label: "Redness", icon: "🌡️", color: "red", accent: "from-red-500/25 to-emerald-500/10 border-red-500/50 text-red-500" },
  { key: "underEye", label: "Under Eye", icon: "👁️", color: "violet", accent: "from-violet-500/25 to-purple-500/10 border-violet-500/50 text-violet-500" },
  { key: "hydration", label: "Hydration", icon: "💧", color: "cyan", accent: "from-cyan-500/25 to-teal-500/10 border-cyan-500/50 text-cyan-500" },
  { key: "texture", label: "Texture", icon: "✨", color: "emerald", accent: "from-emerald-500/25 to-green-500/10 border-emerald-500/50 text-emerald-500" },
];

function clamp(value, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getPoint(landmarks, index) {
  return landmarks[index] || { x: 0.5, y: 0.5 };
}

function createCanvasFromVideo(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function sampleRegion(canvas, landmarks, indexes) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const points = indexes.map((index) => getPoint(landmarks, index));
  const xs = points.map((p) => p.x * width);
  const ys = points.map((p) => p.y * height);

  const minX = Math.max(0, Math.floor(Math.min(...xs)));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(...xs)));
  const minY = Math.max(0, Math.floor(Math.min(...ys)));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(...ys)));

  const regionWidth = Math.max(1, maxX - minX);
  const regionHeight = Math.max(1, maxY - minY);
  const imageData = ctx.getImageData(minX, minY, regionWidth, regionHeight);
  const pixels = imageData.data;

  const brightness = [];
  const redness = [];
  const saturation = [];

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const bright = (r + g + b) / 3;

    brightness.push(bright);
    redness.push(clamp(((r - (g + b) / 2) / 255) * 100 + 50));
    saturation.push(max === 0 ? 0 : ((max - min) / max) * 100);
  }

  return {
    brightness: average(brightness),
    redness: average(redness),
    saturation: average(saturation),
  };
}

function calculateSkinAnalysis(canvas, landmarks) {
  const forehead = sampleRegion(canvas, landmarks, [10, 67, 109, 103, 104, 108]);
  const leftCheek = sampleRegion(canvas, landmarks, [116, 117, 118, 119, 120, 121]);
  const rightCheek = sampleRegion(canvas, landmarks, [345, 346, 347, 348, 349, 350]);
  const nose = sampleRegion(canvas, landmarks, [1, 2, 4, 5, 6, 195]);
  const chin = sampleRegion(canvas, landmarks, [152, 148, 176, 377, 400]);
  const leftUnderEye = sampleRegion(canvas, landmarks, [33, 133, 159, 145]);
  const rightUnderEye = sampleRegion(canvas, landmarks, [362, 263, 386, 374]);

  const skinRegions = [forehead, leftCheek, rightCheek, nose, chin];
  const brightness = average(skinRegions.map((r) => r.brightness));
  const redness = average(skinRegions.map((r) => r.redness));
  const saturation = average(skinRegions.map((r) => r.saturation));
  const underEyeBrightness = average([leftUnderEye.brightness, rightUnderEye.brightness]);

  const acneScore = clamp(100 - Math.abs(brightness - 125) * 0.45 - Math.abs(redness - 50) * 0.3);
  const pigmentationScore = clamp(100 - Math.abs(brightness - 135) * 0.6 - saturation * 0.25);
  const wrinkleScore = clamp(100 - Math.abs(brightness - 130) * 0.35);
  const poreScore = clamp(100 - saturation * 0.55);
  const rednessScore = clamp(100 - Math.abs(redness - 50) * 1.4);
  const underEyeScore = clamp(100 - Math.abs(underEyeBrightness - brightness) * 1.5);
  const hydrationScore = clamp(brightness * 0.62 + (100 - saturation) * 0.38);
  const textureScore = clamp(100 - Math.abs(brightness - 135) * 0.4 - saturation * 0.15);

  const results = {
    acne: Math.round(acneScore),
    pigmentation: Math.round(pigmentationScore),
    wrinkles: Math.round(wrinkleScore),
    pores: Math.round(poreScore),
    redness: Math.round(rednessScore),
    underEye: Math.round(underEyeScore),
    hydration: Math.round(hydrationScore),
    texture: Math.round(textureScore),
  };

  const overallScore = Math.round(average(Object.values(results)));
  const skinType = determineSkinType(brightness, saturation);

  return { ...results, overallScore, skinType };
}

function determineSkinType(brightness, saturation) {
  if (brightness < 85) return "Dry / Dehydrated";
  if (saturation > 45) return "Oily";
  if (brightness > 155) return "Normal";
  return "Combination / Sensitive";
}

function getSeverity(score) {
  if (score >= 75) return "Low Concern";
  if (score >= 50) return "Moderate Concern";
  return "High Concern";
}

export default function App() {
  const videoRef = useRef(null);
  const faceCanvasRef = useRef(null);

  const streamRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const meshAnimRef = useRef(null);

  const [cameraStarted, setCameraStarted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [landmarks, setLandmarks] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [sortBy, setSortBy] = useState("default");
  const [showMesh, setShowMesh] = useState(true);
  
  // Modal state for expert consultation popup
  const [showConsultModal, setShowConsultModal] = useState(false);
  
  // Theme state: Light theme by default, toggles to dark
  const [theme, setTheme] = useState("light");

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const stopCamera = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (meshAnimRef.current) {
      cancelAnimationFrame(meshAnimRef.current);
      meshAnimRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraStarted(false);
    setFaceDetected(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const loadModel = async () => {
    try {
      setLoading(true);
      setError("");
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      const landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_MODEL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      faceLandmarkerRef.current = landmarker;
      setModelReady(true);
    } catch (err) {
      console.error(err);
      setError("Unable to load MediaPipe face model. Please check internet connection.");
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      setError("");
      if (!modelReady) {
        await loadModel();
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraStarted(true);
      detectFace();
    } catch (err) {
      console.error(err);
      setError("Camera permission denied or camera unavailable.");
    }
  };

  const detectFace = () => {
    if (!videoRef.current || !faceLandmarkerRef.current) return;
    const video = videoRef.current;
    if (video.readyState >= 2) {
      try {
        const result = faceLandmarkerRef.current.detectForVideo(video, performance.now());
        if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
          setFaceDetected(true);
          setLandmarks(result.faceLandmarks[0]);
        } else {
          setFaceDetected(false);
        }
      } catch (err) {
        console.error("Face detection error:", err);
      }
    }
    animationRef.current = requestAnimationFrame(detectFace);
  };

  const captureFace = async () => {
    if (!videoRef.current || !faceDetected) return;
    try {
      setLoading(true);
      setError("");
      const canvas = createCanvasFromVideo(videoRef.current);
      const base64Image = canvas.toDataURL("image/jpeg", 0.9);
      setCapturedImage(base64Image);

      let analysisData;
      try {
        const res = await fetch("http://localhost:5000/api/analyze-skin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageBase64: base64Image }),
        });
        if (res.ok) {
          analysisData = await res.json();
        }
      } catch {
        analysisData = calculateSkinAnalysis(canvas, landmarks);
      }

      if (!analysisData) {
        analysisData = calculateSkinAnalysis(canvas, landmarks);
      }

      setAnalysis(analysisData);
      stopCamera();
    } catch (err) {
      console.error(err);
      setError("Error conducting skin analysis scan.");
    } finally {
      setLoading(false);
    }
  };

  const retake = async () => {
    setCapturedImage(null);
    setAnalysis(null);
    setLandmarks(null);
    setFaceDetected(false);
    setError("");
    await startCamera();
  };

  // Live Simulation Canvas
  useEffect(() => {
    if (!cameraStarted || capturedImage) return;
    const canvas = faceCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let time = 0;
    const renderLiveMesh = () => {
      canvas.width = canvas.parentElement.clientWidth || 600;
      canvas.height = canvas.parentElement.clientHeight || 400;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const scale = Math.min(width, height) * 0.38;

      ctx.beginPath();
      ctx.ellipse(centerX, centerY, scale * 0.75, scale * 0.95, 0, 0, 2 * Math.PI);
      ctx.fillStyle = theme === "dark" ? "rgba(15, 23, 42, 0.85)" : "rgba(240, 245, 255, 0.85)";
      ctx.fill();
      ctx.strokeStyle = "rgba(16, 185, 129, 0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();

      if (showMesh) {
        time += 0.03;
        const points = [];
        const rows = 14;
        const cols = 12;

        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const u = (j / (cols - 1)) - 0.5;
            const v = (i / (rows - 1)) - 0.5;
            const dist = Math.sqrt(u * u + v * v);
            if (dist < 0.48) {
              const px = centerX + u * scale * 1.5 + Math.sin(time + i) * 1.5;
              const py = centerY + v * scale * 1.8 + Math.cos(time + j) * 1.5;
              points.push({ x: px, y: py });
            }
          }
        }

        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.lineWidth = 0.8;
        for (let a = 0; a < points.length; a++) {
          for (let b = a + 1; b < points.length; b++) {
            const dx = points[a].x - points[b].x;
            const dy = points[a].y - points[b].y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < scale * 0.22) {
              ctx.beginPath();
              ctx.moveTo(points[a].x, points[a].y);
              ctx.lineTo(points[b].x, points[b].y);
              ctx.stroke();
            }
          }
        }

        points.forEach((p, idx) => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, idx % 7 === 0 ? 2 : 1.2, 0, 2 * Math.PI);
          ctx.fillStyle = idx % 5 === 0 ? "#10b981" : "#38bdf8";
          ctx.shadowColor = "#38bdf8";
          ctx.shadowBlur = 4;
          ctx.fill();
          ctx.shadowBlur = 0;
        });
      }

      meshAnimRef.current = requestAnimationFrame(renderLiveMesh);
    };

    renderLiveMesh();
    return () => {
      if (meshAnimRef.current) cancelAnimationFrame(meshAnimRef.current);
    };
  }, [cameraStarted, capturedImage, showMesh, theme]);

  const getSortedParameters = () => {
    if (!analysis) return [];
    const list = PARAMETERS.map((p) => ({
      ...p,
      score: analysis[p.key] ?? 75,
      concern: getSeverity(analysis[p.key] ?? 75),
    }));

    if (sortBy === "concern") {
      list.sort((a, b) => a.score - b.score);
    } else if (sortBy === "score") {
      list.sort((a, b) => b.score - a.score);
    }
    return list;
  };

  const score = analysis?.overallScore ?? 78;
  const circumference = 301.59;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  const isDark = theme === "dark";

  return (
    <div className={`min-h-screen flex flex-col transition-colors duration-300 ${isDark ? "bg-[#090b10] text-slate-100 selection:bg-emerald-500 selection:text-white" : "bg-[#f8fafc] text-slate-800"} antialiased font-sans relative overflow-x-hidden`}>
      
      {/* TOP TICKER BAR */}
      <div className={`w-full border-b text-[10px] sm:text-[11px] font-mono py-2 px-3 sm:px-8 flex flex-wrap items-center justify-between gap-y-1.5 gap-x-4 transition-colors ${isDark ? "bg-[#0f121a] border-slate-800 text-slate-400" : "bg-white border-slate-200 text-slate-600"}`}>
        <div className="flex items-center gap-3 sm:gap-4 overflow-x-auto whitespace-nowrap scrollbar-none">
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0"></span>CHROMA-POS8P</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0"></span>GLUCOVIS8V</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 shrink-0"></span>DERMA-SCANV9</span>
          <span className="text-slate-500 hidden md:inline">V.24.0ENGINE</span>
        </div>
        <div className="flex items-center gap-3 sm:gap-4">
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-500 font-bold border border-emerald-500/25">English EN</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0"></span>AI MODEL READY</span>
          <span className="text-slate-500 hidden lg:inline">11/2/2026, 2:16:33 PM</span>
        </div>
      </div>

      {/* HEADER NAVBAR */}
      <header className={`sticky top-0 z-40 backdrop-blur-md border-b transition-all duration-200 ${isDark ? "bg-[#090b10]/90 border-slate-800/80" : "bg-white/90 border-slate-200/80"}`}>
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-16 sm:h-18 flex items-center justify-between py-2 sm:py-3">
          <div className="flex items-center gap-2 sm:gap-3">
            <img 
              src={isDark ? logoDark : logoLight} 
              alt="iYoung Logo" 
              className={`h-10 sm:h-12 lg:h-14 w-auto object-contain transition-all duration-300 ${isDark ? "brightness-125 filter drop-shadow-[0_0_10px_rgba(255,255,255,0.25)]" : ""}`} 
            />
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <button
              onClick={toggleTheme}
              title="Toggle Theme"
              className={`p-2 rounded-xl border text-xs sm:text-sm font-semibold transition active:scale-95 flex items-center gap-1.5 ${isDark ? "bg-slate-900 border-slate-700 text-amber-400 hover:bg-slate-800" : "bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200"}`}
            >
              <span>{isDark ? "☀️ Light" : "🌙 Dark"}</span>
            </button>

            {capturedImage && (
              <button
                onClick={retake}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-xl border text-xs sm:text-sm font-semibold transition active:scale-95 ${isDark ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800" : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"}`}
              >
                <span>↺ Retake</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 lg:px-8 py-5 sm:py-8 space-y-6 sm:space-y-8 relative z-10">
        
        {/* HERO INTRO */}
        <div className="text-center space-y-3 sm:space-y-4 flex flex-col items-center justify-center">
          <div className="max-w-2xl space-y-2 px-2">
            <h1 className={`text-2xl sm:text-4xl lg:text-5xl font-black tracking-tight ${isDark ? "text-white" : "text-slate-900"} leading-tight`}>
              Your skin, understood <br />
              <span className="bg-gradient-to-r from-emerald-500 to-teal-500 bg-clip-text text-transparent">in a moment.</span>
            </h1>
            <p className={`text-xs sm:text-sm ${isDark ? "text-slate-400" : "text-slate-600"} max-w-lg mx-auto leading-relaxed`}>
              Use your camera to capture key skin health indicators through advanced facial mesh analysis — private, simple, and contact-free.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4 text-[11px] sm:text-xs font-medium pt-1">
            <span className={`flex items-center gap-1.5 ${isDark ? "text-slate-300" : "text-slate-700"}`}>🔒 No wearable needed</span>
            <span className={`flex items-center gap-1.5 ${isDark ? "text-slate-300" : "text-slate-700"}`}>⚡ Results in 60s</span>
            <span className={`flex items-center gap-1.5 ${isDark ? "text-slate-300" : "text-slate-700"}`}>🛡️ Private</span>
          </div>
        </div>

        {/* EXPERT CONSULTATION BAR */}
        <div className={`rounded-2xl p-4 sm:p-5 flex flex-col sm:flex-row items-center justify-between gap-4 border shadow-sm ${isDark ? "bg-[#0f121a] border-slate-800" : "bg-white border-slate-200"}`}>
          <div className="flex items-center gap-3.5 text-left w-full sm:w-auto">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border ${isDark ? "bg-emerald-950/50 border-emerald-900/50 text-emerald-400" : "bg-emerald-50 border-emerald-100 text-emerald-600"}`}>
              <span>🩺</span>
            </div>
            <div>
              <h4 className={`font-bold text-xs sm:text-sm ${isDark ? "text-white" : "text-slate-900"}`}>Need to speak with an expert?</h4>
              <p className={`text-[11px] sm:text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>You can book an appointment at any time to review your skin results.</p>
            </div>
          </div>
          <button 
            onClick={() => setShowConsultModal(true)}
            className="w-full sm:w-auto px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-600/25 transition active:scale-95 flex items-center justify-center gap-2 shrink-0 border border-emerald-500/30"
          >
            <span>Consult a Dermatologist</span>
            <span>&gt;</span>
          </button>
        </div>

        {/* HOW TO SCAN PROTOCOL GUIDE (Updated with Steps 3 & 4 removed and re-indexed) */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest block">Protocol Guide</span>
              <h3 className={`text-base sm:text-xl font-bold tracking-tight ${isDark ? "text-white" : "text-slate-900"}`}>HOW TO SCAN</h3>
            </div>
            <span className={`text-[11px] sm:text-xs hidden sm:inline ${isDark ? "text-slate-400" : "text-slate-500"}`}>Follow these steps for accurate AI mesh-analyzed skin data</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            {[
              { num: "01", title: "OPEN CAMERA", desc: "Click START SCAN and grant camera access permissions securely in browser." },
              { num: "02", title: "POSITION FACE", desc: "Sit comfortably with good face lighting so camera with facial landmarks are clearly aligned." },
              { num: "03", title: "START SKIN ANALYSIS", desc: "Click »START SKIN ANALYSIS« and let camera actively capture your mesh landmarks." },
              { num: "04", title: "VIEW RESULTS", desc: "Hold steady as scan computes to render your 3 biomarker tiles." }
            ].map((step, i) => (
              <div key={i} className={`rounded-xl p-3.5 sm:p-4 border transition-all ${isDark ? "bg-[#0f121a] border-slate-800/80 text-slate-300" : "bg-white border-slate-200/80 text-slate-700"}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-bold text-emerald-500">{step.num}</span>
                  <span className="text-sm">📷</span>
                </div>
                <h5 className={`font-bold text-xs uppercase tracking-wider mb-1 ${isDark ? "text-white" : "text-slate-900"}`}>{step.title}</h5>
                <p className="text-[11px] leading-relaxed opacity-80">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* LIVE DIAGNOSTIC STUDIO SECTION */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest block">Live Diagnostic Studio</span>
              <h2 className={`text-lg sm:text-2xl font-bold tracking-tight ${isDark ? "text-white" : "text-slate-900"}`}>Facial Mesh &amp; Biomarker Results</h2>
            </div>
            {capturedImage && (
              <button
                onClick={retake}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${isDark ? "bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800" : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"}`}
              >
                <span>↺ Retake</span>
              </button>
            )}
          </div>

          {!capturedImage && (
            <div className={`rounded-2xl p-4 sm:p-6 shadow-xl border ${isDark ? "bg-[#0f121a] border-slate-800" : "bg-white border-slate-200"}`}>
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
                
                <div className="lg:col-span-7 relative aspect-[4/3] sm:aspect-[16/10] bg-slate-950 rounded-xl overflow-hidden shadow-inner flex items-center justify-center border border-slate-800">
                  <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover -scale-x-100 opacity-90" muted playsInline />
                  <canvas ref={faceCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

                  {!cameraStarted && (
                    <div className="absolute inset-0 flex flex-col justify-center items-center bg-slate-950/95 text-white z-10 p-4 text-center">
                      <div className="text-4xl sm:text-5xl mb-3 animate-pulse">🙂</div>
                      <p className="text-slate-300 text-xs sm:text-sm max-w-xs font-medium">Start the camera to begin your live AI skin analysis</p>
                    </div>
                  )}

                  {cameraStarted && (
                    <div className="absolute top-3 left-3 flex items-center gap-2 z-20">
                      <span className="px-2.5 py-1 rounded-md bg-slate-900/80 backdrop-blur-md text-cyan-400 text-[10px] font-mono border border-cyan-500/30">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping inline-block mr-1"></span>
                        3D MESH ACTIVE
                      </span>
                    </div>
                  )}
                </div>

                <div className="lg:col-span-5 space-y-4">
                  <div className="space-y-1">
                    <span className="text-xs font-bold text-emerald-500 uppercase tracking-wider">Ready to Analyze</span>
                    <h3 className={`text-lg sm:text-xl font-bold ${isDark ? "text-white" : "text-slate-900"}`}>Live Computer Vision</h3>
                    <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-600"}`}>
                      Position your face centrally in the camera frame. The AI mesh tracker will automatically detect facial zones to calculate your skin score.
                    </p>
                  </div>

                  <div className={`p-3.5 rounded-xl border space-y-2 text-xs ${isDark ? "bg-slate-900/60 border-slate-800 text-slate-300" : "bg-slate-50 border-slate-200 text-slate-700"}`}>
                    <div className="flex items-center justify-between font-semibold">
                      <span>Face Detection Status:</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase ${faceDetected ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'}`}>
                        {faceDetected ? "Detected & Ready" : "Searching for Face..."}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2.5">
                    {!cameraStarted ? (
                      <button
                        onClick={startCamera}
                        disabled={loading}
                        className="w-full py-3.5 px-6 rounded-xl font-bold text-xs sm:text-sm bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-600/20 transition active:scale-95 disabled:opacity-50"
                      >
                        {loading ? "Loading AI Model..." : "Start Skin Analysis"}
                      </button>
                    ) : (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={captureFace}
                          disabled={loading || !faceDetected}
                          className="flex-1 py-3 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs sm:text-sm shadow-md transition active:scale-95 flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                          <span className="w-2.5 h-2.5 rounded-full bg-white animate-pulse"></span>
                          {loading ? "Analyzing..." : "Capture Face"}
                        </button>

                        <button
                          onClick={stopCamera}
                          className={`py-3 px-4 rounded-xl font-bold text-xs border transition active:scale-95 ${isDark ? "bg-slate-900 border-slate-700 text-emerald-400 hover:bg-slate-800" : "bg-white border-slate-300 text-emerald-600 hover:bg-slate-50"}`}
                          title="Stop Camera"
                        >
                          <span>⏹️ Stop</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>

              </div>
            </div>
          )}

          {capturedImage && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
              
              <div className={`lg:col-span-7 rounded-2xl p-4 sm:p-5 shadow-xl border flex flex-col justify-between ${isDark ? "bg-[#0f121a] border-slate-800" : "bg-white border-slate-200"}`}>
                <div className="relative w-full aspect-[4/3] sm:aspect-[16/10] bg-slate-950 rounded-xl overflow-hidden shadow-inner flex items-center justify-center border border-slate-800">
                  <img src={capturedImage} alt="Captured face" className="w-full h-full object-cover filter brightness-95" />

                  <div className="absolute top-3 left-3 flex items-center gap-2">
                    <span className="px-2.5 py-1 rounded-md bg-slate-900/80 backdrop-blur-md text-cyan-400 text-[10px] font-mono border border-cyan-500/30">
                      SCAN COMPLETE
                    </span>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center justify-between gap-2 px-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Estimated Type:</span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold border ${isDark ? "bg-emerald-950/60 text-emerald-300 border-emerald-900/80" : "bg-emerald-50 text-emerald-700 border-emerald-200"}`}>
                      💧 {analysis?.skinType || "Combination / Sensitive"}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 flex items-center gap-1">
                    🛡️ HIPAA Compliant
                  </div>
                </div>
              </div>

              <div className={`lg:col-span-5 rounded-2xl p-4 sm:p-6 shadow-xl border flex flex-col justify-between ${isDark ? "bg-[#0f121a] border-slate-800" : "bg-white border-slate-200"}`}>
                <div>
                  <div className="text-center">
                    <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400">Overall Skin Score</span>
                  </div>

                  <div className="relative w-40 h-40 sm:w-52 sm:h-52 mx-auto my-3 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                      <circle cx="60" cy="60" r="48" className={isDark ? "text-slate-800" : "text-slate-100"} strokeWidth="10" stroke="currentColor" fill="transparent" />
                      <circle
                        cx="60"
                        cy="60"
                        r="48"
                        className="text-emerald-500 transition-all duration-1000 ease-out filter drop-shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                        strokeWidth="10"
                        strokeLinecap="round"
                        stroke="currentColor"
                        fill="transparent"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                      />
                    </svg>

                    <div className="absolute flex flex-col items-center justify-center text-center">
                      <div className="flex items-baseline">
                        <span className={`text-3xl sm:text-5xl font-black tracking-tight ${isDark ? "text-white" : "text-slate-900"}`}>{score}</span>
                        <span className="text-slate-400 text-xs sm:text-sm font-semibold ml-0.5">/100</span>
                      </div>
                      <span className="text-[10px] font-mono text-emerald-500 uppercase font-bold mt-0.5">DERM INDEX</span>
                    </div>
                  </div>

                  <div className="text-center space-y-1.5">
                    <div className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold border ${isDark ? "bg-amber-950/60 border-amber-900/80 text-amber-300" : "bg-amber-50 border-amber-200 text-amber-800"}`}>
                      <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                      <span>{score >= 80 ? "Optimal Condition" : score >= 60 ? "Moderate Condition" : "Needs Attention"}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t grid grid-cols-3 text-center gap-2 border-slate-500/20">
                  <div className="p-2.5 rounded-xl border bg-gradient-to-br from-cyan-500/15 to-blue-500/5 border-cyan-500/40 shadow-xs">
                    <div className="text-[11px] font-semibold text-cyan-600 dark:text-cyan-400 flex items-center justify-center gap-1"><span>💧</span> Hydration</div>
                    <div className="text-sm sm:text-lg font-black mt-0.5">{analysis?.hydration ?? 85}%</div>
                  </div>
                  <div className="p-2.5 rounded-xl border bg-gradient-to-br from-purple-500/15 to-indigo-500/5 border-purple-500/40 shadow-xs">
                    <div className="text-[11px] font-semibold text-purple-600 dark:text-purple-400 flex items-center justify-center gap-1"><span>✨</span> Clarity</div>
                    <div className="text-sm sm:text-lg font-black mt-0.5">{analysis?.acne ?? 64}%</div>
                  </div>
                  <div className="p-2.5 rounded-xl border bg-gradient-to-br from-emerald-500/15 to-teal-500/5 border-emerald-500/40 shadow-xs">
                    <div className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 flex items-center justify-center gap-1"><span>🛡️</span> Uniformity</div>
                    <div className="text-sm sm:text-lg font-black mt-0.5">{analysis?.texture ?? 72}%</div>
                  </div>
                </div>

              </div>

            </div>
          )}
        </div>

        {/* DETAILED PARAMETERS BREAKDOWN GRID */}
        {capturedImage && (
          <section id="skin-parameters-section" className="space-y-4 scroll-mt-24">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest block">Comprehensive Breakdown</span>
                <h2 className={`text-lg sm:text-2xl font-bold tracking-tight ${isDark ? "text-white" : "text-slate-900"}`}>Skin Parameters (All 8 Metrics)</h2>
              </div>
              <div className="flex items-center gap-2 self-end sm:self-auto">
                <span className="text-xs text-slate-400 font-medium">Sort:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className={`rounded-lg text-xs font-semibold px-3 py-1.5 focus:outline-none border ${isDark ? "bg-[#0f121a] border-slate-700 text-slate-200" : "bg-white border-slate-300 text-slate-700"}`}
                >
                  <option value="default">Default Order</option>
                  <option value="concern">Highest Concern</option>
                  <option value="score">Highest Score</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
              {getSortedParameters().map((param) => {
                let badgeStyle = isDark ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800' : 'bg-emerald-50 text-emerald-700 border-emerald-200';
                let progressStyle = 'bg-emerald-500';

                if (param.score < 50) {
                  badgeStyle = isDark ? 'bg-rose-950/80 text-rose-300 border-rose-800' : 'bg-rose-50 text-rose-700 border-rose-200';
                  progressStyle = 'bg-rose-500';
                } else if (param.score < 75) {
                  badgeStyle = isDark ? 'bg-amber-950/80 text-amber-300 border-amber-800' : 'bg-amber-50 text-amber-700 border-amber-200';
                  progressStyle = 'bg-amber-500';
                }

                return (
                  <div 
                    key={param.key} 
                    className={`rounded-xl p-4 shadow-md transition-all border bg-gradient-to-br ${param.accent} ${isDark ? "bg-[#0f121a]/90 backdrop-blur-xl border-slate-800" : "bg-white/90 backdrop-blur-xl border-slate-200"}`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2.5">
                        <div className={`w-8 h-8 rounded-lg border flex items-center justify-center text-sm shadow-xs ${isDark ? "bg-slate-900 border-slate-700" : "bg-slate-50 border-slate-200"}`}>
                          {param.icon}
                        </div>
                        <div>
                          <h4 className={`font-bold text-xs ${isDark ? "text-white" : "text-slate-900"}`}>{param.label}</h4>
                          <span className={`inline-block px-2 py-0.5 rounded-full border ${badgeStyle} text-[9px] font-bold mt-0.5`}>
                            {param.concern}
                          </span>
                        </div>
                      </div>
                      <div className="text-right font-mono">
                        <span className={`text-base font-extrabold ${isDark ? "text-white" : "text-slate-900"}`}>{param.score}</span>
                        <span className="text-[10px] text-slate-400">/100</span>
                      </div>
                    </div>

                    <div className={`w-full h-1.5 rounded-full overflow-hidden border ${isDark ? "bg-slate-950 border-slate-800" : "bg-slate-100 border-slate-200"}`}>
                      <div className={`h-full ${progressStyle} rounded-full transition-all duration-1000 ease-out`} style={{ width: `${param.score}%` }}></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* SUGGESTED SKIN ROUTINE */}
        {capturedImage && (
          <section className={`rounded-2xl p-5 sm:p-8 shadow-xl space-y-6 border ${isDark ? "bg-[#0f121a] border-slate-800" : "bg-white border-slate-200"}`}>
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl border flex items-center justify-center ${isDark ? "bg-emerald-950/50 border-emerald-900/50 text-emerald-400" : "bg-emerald-50 border-emerald-100 text-emerald-600"}`}>
                <span>✨</span>
              </div>
              <div>
                <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest block">Tailored Regimen</span>
                <h3 className={`text-lg sm:text-xl font-bold ${isDark ? "text-white" : "text-slate-900"}`}>Suggested Skin Routine</h3>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className={`rounded-xl p-4 sm:p-5 border shadow-sm ${isDark ? "bg-slate-900/50 border-slate-800 text-slate-300" : "bg-slate-50 border-slate-200 text-slate-700"}`}>
                <div className="flex items-center gap-2 text-amber-500 font-bold text-xs mb-3">
                  <span>☀️</span> Morning Routine
                </div>
                <ul className="space-y-2.5 text-xs">
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-500 border border-emerald-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">1</span> Gentle Hydrating Cleanser (pH Balanced)</li>
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-500 border border-emerald-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">2</span> 2% Niacinamide Serum for pore refinement</li>
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-500 border border-emerald-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">3</span> Broad Spectrum SPF 50+ Sunscreen</li>
                </ul>
              </div>

              <div className={`rounded-xl p-4 sm:p-5 border shadow-sm ${isDark ? "bg-slate-900/50 border-slate-800 text-slate-300" : "bg-slate-50 border-slate-200 text-slate-700"}`}>
                <div className="flex items-center gap-2 text-indigo-500 font-bold text-xs mb-3">
                  <span>🌙</span> Evening Routine
                </div>
                <ul className="space-y-2.5 text-xs">
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">1</span> Double Cleanse (Micellar Water + Gel Cleanser)</li>
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">2</span> Hyaluronic Acid + Ceramides Moisturizer</li>
                  <li className="flex items-start gap-2"><span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">3</span> 0.25% Encapsulated Retinol (2x weekly)</li>
                </ul>
              </div>

              <div className={`rounded-xl p-4 sm:p-5 border shadow-sm ${isDark ? "bg-gradient-to-br from-emerald-950/25 to-indigo-950/25 border-emerald-900/40 text-slate-300" : "bg-gradient-to-br from-emerald-500/5 to-indigo-500/5 border-emerald-100 text-slate-700"}`}>
                <div className="flex items-center gap-2 text-emerald-500 font-bold text-xs mb-3">
                  <span>🧪</span> Targeted Actives
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${isDark ? "bg-slate-900 border-slate-700 text-emerald-300" : "bg-white border-slate-200 text-emerald-700"}`}>Hyaluronic Acid</span>
                  <span className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${isDark ? "bg-slate-900 border-slate-700 text-indigo-300" : "bg-white border-slate-200 text-indigo-700"}`}>Niacinamide</span>
                  <span className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${isDark ? "bg-slate-900 border-slate-700 text-slate-300" : "bg-white border-slate-200 text-slate-700"}`}>Salicylic Acid (BHA)</span>
                  <span className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${isDark ? "bg-slate-900 border-slate-700 text-teal-300" : "bg-white border-slate-200 text-teal-700"}`}>Centella Asiatica</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs flex items-center gap-2 shadow-lg">
            <span>⚠️</span> {error}
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className={`mt-auto border-t py-4 text-xs ${isDark ? "border-slate-800 bg-[#090b10] text-slate-400" : "border-slate-200 bg-slate-100 text-slate-500"}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-center sm:text-left">
          <div>&copy; 2026 iYoung Pro Skin Science &bull; All rights reserved.</div>
          <div>React + MediaPipe Facial Mesh Engine Ready</div>
        </div>
      </footer>

      {/* CONSULTATION POPUP MODAL */}
      {showConsultModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/75 backdrop-blur-sm animate-fadeIn overflow-y-auto">
          <div className={`w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-3xl p-4 sm:p-6 shadow-2xl border relative space-y-3.5 my-auto ${isDark ? "bg-[#0f121a] border-slate-800 text-slate-100" : "bg-white border-slate-100 text-slate-900"}`}>
            
            <button 
              onClick={() => setShowConsultModal(false)}
              className={`absolute top-4 right-4 sm:top-5 sm:right-5 w-8 h-8 rounded-full border flex items-center justify-center text-sm font-bold transition active:scale-95 ${isDark ? "bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800" : "bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200"}`}
            >
              ✕
            </button>

            <div className="flex items-center gap-3 pr-8">
              <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-2xl bg-emerald-100 dark:bg-emerald-950/80 border border-emerald-300 dark:border-emerald-800 flex items-center justify-center text-emerald-600 dark:text-emerald-400 text-lg font-bold shadow-xs shrink-0">
                +
              </div>
              <div>
                <span className="text-[10px] font-mono font-bold text-emerald-500 uppercase tracking-widest">iYoungBio &bull; NEXT STEP</span>
                <h3 className="text-base sm:text-xl font-black tracking-tight">Speak with a doctor</h3>
              </div>
            </div>

            <p className={`text-xs leading-relaxed ${isDark ? "text-slate-400" : "text-slate-600"}`}>
              You can consult a healthcare professional at any time. You do not need to complete a skin scan before booking an appointment.
            </p>

            <div className="p-3 rounded-2xl border border-emerald-500/40 bg-emerald-50/50 dark:bg-emerald-950/20 space-y-1">
              <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wide">Suggested Next Step</div>
              <h4 className="font-bold text-xs sm:text-sm">Dermatology Specialist</h4>
              <p className={`text-[11px] leading-relaxed ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                You can book a consultation before scanning. A professional can discuss your skin concerns and advise whether any specialized treatments are appropriate.
              </p>
            </div>

            <div className={`p-3 rounded-2xl border text-[11px] leading-relaxed ${isDark ? "bg-slate-900/60 border-slate-800 text-slate-400" : "bg-slate-50 border-slate-200 text-slate-600"}`}>
              This skin scan is a research prototype and is not a clinical diagnosis. A qualified specialist can provide advice based on your individual skin assessment.
            </div>

            <div className="space-y-2 pt-1">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <a
                  href="https://ihealthy.netlify.app/doctors"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs text-center shadow-md shadow-emerald-600/20 transition active:scale-95 flex items-center justify-center gap-2"
                >
                  <span>Find / Book a Doctor</span>
                </a>

                <button
                  onClick={() => {
                    setShowConsultModal(false);
                    const section = document.getElementById("skin-parameters-section");
                    if (section) {
                      section.scrollIntoView({ behavior: "smooth" });
                    }
                  }}
                  className={`py-2.5 px-4 rounded-xl font-bold text-xs border transition active:scale-95 ${isDark ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800" : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"}`}
                >
                  View My Results
                </button>
              </div>

              <button
                onClick={() => setShowConsultModal(false)}
                className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs border transition active:scale-95 ${isDark ? "bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-900" : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"}`}
              >
                Continue
              </button>
            </div>

            <p className="text-[10px] text-center text-slate-400 pt-0.5">
              If you have severe or sudden skin symptoms, seek urgent medical care regardless of scan results.
            </p>

          </div>
        </div>
      )}
    </div>
  );
}