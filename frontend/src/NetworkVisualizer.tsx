import React, { useEffect, useMemo, useRef, useState } from "react";

interface Props {
  inputTensor: Float32Array | null;
  activations: number[][]; // [Layer 1, Layer 2]
  probabilities: number[]; // Output Layer
}

export const NetworkVisualizer: React.FC<Props> = ({
  inputTensor,
  activations,
  probabilities,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Maximum number of neurons to draw per layer to not overload the screen
  const MAX_NEURONS = 16;
  const INPUT_VECTOR_SIZE = 784;

  const inputPreview = useMemo(() => {
    if (!inputTensor || inputTensor.length === 0) return [];

    const visibleInput = Math.min(MAX_NEURONS, inputTensor.length);
    const sampled: number[] = [];
    for (let i = 0; i < visibleInput; i++) {
      const idx = Math.floor(
        (i / Math.max(visibleInput - 1, 1)) * (inputTensor.length - 1),
      );
      sampled.push(inputTensor[idx] ?? 0);
    }
    return sampled;
  }, [inputTensor]);

  // Input + hidden layers + output
  const allLayers = useMemo(
    () => [inputPreview, ...activations, probabilities],
    [inputPreview, activations, probabilities],
  );

  const layerTitles = useMemo(() => {
    return allLayers.map((_layer, layerIndex) => {
      if (layerIndex === 0) return `Input (${INPUT_VECTOR_SIZE})`;
      if (layerIndex === allLayers.length - 1) return "Output (10)";
      return `Hidden ${layerIndex} (${activations[layerIndex - 1]?.length ?? 0})`;
    });
  }, [allLayers, activations]);

  // State to force the re-render and recalculate the lines if the window is resized
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };
    window.addEventListener("resize", updateSize);
    updateSize();
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  // This Effect draws the SYNAPSES (lines)
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || dimensions.width === 0 || !containerRef.current)
      return;

    // Adapt the canvas to the container for high definition
    canvas.width = dimensions.width * window.devicePixelRatio;
    canvas.height = dimensions.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    const containerRect = containerRef.current.getBoundingClientRect();
    const getNeuronCenter = (layerIdx: number, neuronIdx: number) => {
      const el = containerRef.current?.querySelector<HTMLElement>(
        `[data-neuron="${layerIdx}-${neuronIdx}"]`,
      );
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      return {
        x: rect.left - containerRect.left + rect.width / 2,
        y: rect.top - containerRect.top + rect.height / 2,
      };
    };

    const layerCount = allLayers.length;

    // Draw the lines between the Layer i and the Layer i+1
    for (let l = 0; l < layerCount - 1; l++) {
      const currentLayer = allLayers[l];
      const nextLayer = allLayers[l + 1];

      const visibleCurrent = Math.min(currentLayer.length, MAX_NEURONS);
      const visibleNext = Math.min(nextLayer.length, MAX_NEURONS);

      for (let i = 0; i < visibleCurrent; i++) {
        for (let j = 0; j < visibleNext; j++) {
          const actStart = currentLayer[i];
          const actEnd = nextLayer[j];

          // The line only illuminates if BOTH neurons are active
          const intensity = actStart * actEnd;

          if (intensity > 0.05) {
            // The coordinates are read from the real neurons in the DOM, so lines and nodes remain aligned.
            const start = getNeuronCenter(l, i);
            const end = getNeuronCenter(l + 1, j);
            if (!start || !end) continue;

            // Draw the line
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);

            // We use Bezier curves to give a "organic" look instead of sharp edges
            const controlX = start.x + (end.x - start.x) * 0.5;
            ctx.bezierCurveTo(controlX, start.y, controlX, end.y, end.x, end.y);

            // Thin strokes: many edges to the same neuron stack visually
            const strokeAlpha = intensity * 0.45;
            ctx.strokeStyle = `rgba(34, 211, 238, ${strokeAlpha})`;
            ctx.lineWidth = Math.min(0.55 + intensity * 1.15, 1.85);
            ctx.stroke();
          }
        }
      }
    }
  }, [allLayers, dimensions]);

  return (
    <div className="relative flex w-full min-h-[660px] flex-col bg-slate-900/50 rounded-xl border border-slate-700 shadow-inner">
      {/* Header in the flow: reserve real space so the columns don't go under the title */}
      <div className="shrink-0 border-b border-slate-800/60 px-4 pt-4 pb-3">
        <div className="text-center text-[10px] uppercase tracking-[0.25em] text-slate-500 font-mono">
          Network Layers
        </div>
        <div
          className="mt-2 grid"
          style={{
            gridTemplateColumns: `repeat(${allLayers.length}, minmax(0, 1fr))`,
          }}
        >
          {layerTitles.map((title) => (
            <div
              key={title}
              className="text-center text-[10px] uppercase text-slate-400 font-mono font-bold tracking-widest"
            >
              {title}
            </div>
          ))}
        </div>
      </div>

      {/* ref here: canvas and neuron coordinates share the same bounding box */}
      <div
        ref={containerRef}
        className="relative flex min-h-[560px] flex-1 flex-row items-center justify-around px-4 py-8"
      >
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 z-0 h-full w-full"
        />

        {/* Rendering of the Neurons (DOM Elements) in overlay */}
        {allLayers.map((layer, layerIndex) => {
          const visibleNeurons = layer.slice(0, MAX_NEURONS);
          const isOutputLayer = layerIndex === allLayers.length - 1;

          return (
            <div
              key={`layer-${layerIndex}`}
              className="relative z-10 flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-2"
            >
              <div className="flex flex-col items-center gap-[14px]">
                {visibleNeurons.map((activation, neuronIndex) => (
                  <div
                    key={`n-${layerIndex}-${neuronIndex}`}
                    className="flex items-center gap-3"
                  >
                    <div
                      data-neuron={`${layerIndex}-${neuronIndex}`}
                      className="w-4 h-4 rounded-full bg-cyan-400 transition-all duration-150 relative"
                      style={{
                        opacity: 0.15 + activation * 0.85,
                        boxShadow:
                          activation > 0.3
                            ? `0 0 ${activation * 15}px #22d3ee, inset 0 0 4px white`
                            : "none",
                        transform: `scale(${1 + activation * 0.2})`, // They slightly increase if very active
                      }}
                    />
                    {/* For the last layer we show the number from 0 to 9 next to the neuron */}
                    {isOutputLayer && (
                      <span className="text-xs font-mono font-bold text-slate-300 w-4">
                        {neuronIndex}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
