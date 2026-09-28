import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HistogramBin } from "../api";

interface Props {
  histogram: HistogramBin[];
  medianScore: number;
}

export function ScoreDistributionChart({ histogram, medianScore }: Props) {
  const data = histogram.map((b) => ({
    label: `${b.range_low}-${b.range_high}`,
    mid: (b.range_low + b.range_high) / 2,
    probability: b.probability * 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="mid" tick={{ fontSize: 11 }} label={{ value: "Score", position: "insideBottom", offset: -4 }} />
        <YAxis tick={{ fontSize: 11 }} label={{ value: "Probability (%)", angle: -90, position: "insideLeft" }} />
        <Tooltip
          formatter={(value) => [`${Number(value).toFixed(2)}%`, "Probability"]}
          labelFormatter={(label) => `Score ≈ ${label}`}
        />
        <ReferenceLine x={medianScore} stroke="var(--accent)" strokeDasharray="4 4" label={{ value: "Median", fontSize: 11, fill: "var(--accent)" }} />
        <Bar dataKey="probability" fill="var(--accent)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
