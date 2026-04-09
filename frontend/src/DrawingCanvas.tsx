import React, { useRef, useState, useEffect, useCallback } from "react";

interface DrawingCanvasProps {
  onDrawEnd: (tensor: Float32Array) => void;
  onClear: () => void;
  clearLabel: string;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  onDrawEnd,
  onClear,
  clearLabel,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const offCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  const CANVAS_SIZE = 280;
  const TARGET_SIZE = 28;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return;

    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "white";
    context.lineWidth = 18; // Slightly thicker to compensate for the resize
    contextRef.current = context;

    const offCanvas = document.createElement("canvas");
    offCanvas.width = TARGET_SIZE;
    offCanvas.height = TARGET_SIZE;
    const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });
    offCanvasRef.current = offCanvas;
    offCtxRef.current = offCtx;
  }, []);

  // --- ounding Box Algorithm ---
  const processAndDispatchTensor = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = contextRef.current;
    const offCtx = offCtxRef.current;
    if (!canvas || !ctx || !offCtx) return;

    // 1. Find the boundaries of the drawing (ignoring the black background)
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const data = imgData.data;
    let minX = CANVAS_SIZE,
      minY = CANVAS_SIZE,
      maxX = 0,
      maxY = 0;
    let hasDrawn = false;

    for (let y = 0; y < CANVAS_SIZE; y++) {
      for (let x = 0; x < CANVAS_SIZE; x++) {
        const red = data[(y * CANVAS_SIZE + x) * 4]; // Red channel
        if (red > 10) {
          // If there is color
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
          hasDrawn = true;
        }
      }
    }

    // If the canvas is empty, send an array of zeros
    if (!hasDrawn) {
      onDrawEnd(new Float32Array(TARGET_SIZE * TARGET_SIZE));
      return;
    }

    // Add a little margin to not cut the soft shades of the pen
    const padding = 15;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(CANVAS_SIZE, maxX + padding);
    maxY = Math.min(CANVAS_SIZE, maxY + padding);

    const bboxWidth = maxX - minX;
    const bboxHeight = maxY - minY;

    // 2. MNIST Style: Scale to fit the drawing into a 20x20 square
    const scale = 20.0 / Math.max(bboxWidth, bboxHeight);
    const scaledWidth = bboxWidth * scale;
    const scaledHeight = bboxHeight * scale;

    // 3. Calculate the offset to center it in the final 28x28 canvas
    const dx = (TARGET_SIZE - scaledWidth) / 2;
    const dy = (TARGET_SIZE - scaledHeight) / 2;

    // 4. Draw on the small canvas
    offCtx.fillStyle = "black";
    offCtx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    offCtx.drawImage(
      canvas,
      minX,
      minY,
      bboxWidth,
      bboxHeight, // Crop the Bounding Box...
      dx,
      dy,
      scaledWidth,
      scaledHeight, // ...and paste it scaled and centered
    );

    // 5. Extract and normalize
    const finalImgData = offCtx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
    const finalData = finalImgData.data;
    const tensor = new Float32Array(TARGET_SIZE * TARGET_SIZE);

    for (let i = 0; i < tensor.length; i++) {
      tensor[i] = finalData[i * 4] / 255.0;
    }

    onDrawEnd(tensor);
  }, [onDrawEnd]);

  // --- Event Handling for Mouse/Touch ---
  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    const { offsetX, offsetY } = getCoordinates(e);
    contextRef.current?.beginPath();
    contextRef.current?.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);

    contextRef.current?.lineTo(offsetX, offsetY);
    contextRef.current?.stroke();

    processAndDispatchTensor();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    contextRef.current?.closePath();
    setIsDrawing(false);
    processAndDispatchTensor();
  };

  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { offsetX: 0, offsetY: 0 };
    const rect = canvas.getBoundingClientRect();
    if ("touches" in e) {
      return {
        offsetX: e.touches[0].clientX - rect.left,
        offsetY: e.touches[0].clientY - rect.top,
      };
    }
    return {
      offsetX: e.nativeEvent.offsetX,
      offsetY: e.nativeEvent.offsetY,
    };
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;
    contextRef.current.fillStyle = "black";
    contextRef.current.fillRect(0, 0, canvas.width, canvas.height);
    onClear();
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        className="bg-black border-2 border-slate-600 rounded-lg cursor-crosshair touch-none shadow-xl shadow-cyan-900/20"
        style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
      />
      <button
        onClick={clearCanvas}
        className="px-6 py-2 font-medium text-slate-300 hover:text-white hover:bg-slate-700 bg-slate-800 rounded-lg transition-colors border border-slate-600"
      >
        {clearLabel}
      </button>
    </div>
  );
};
