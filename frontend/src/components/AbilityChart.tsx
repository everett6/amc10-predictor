import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface Props {
  abilityByCategory: Record<string, number>;
  globalAbility: number;
}

export function AbilityChart({ abilityByCategory, globalAbility }: Props) {
  const data = Object.entries(abilityByCategory)
    .map(([category, ability]) => ({ category, ability: Math.round(ability) }))
    .sort((a, b) => b.ability - a.ability);

  return (
    <ResponsiveContainer width="100%" height={Math.max(280, data.length * 28)}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" domain={[0, "dataMax + 100"]} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="category" width={260} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(value) => [value, "Elo-scale ability"]} />
        <Bar dataKey="ability" radius={[0, 3, 3, 0]}>
          {data.map((d) => (
            <Cell key={d.category} fill={d.ability >= globalAbility ? "var(--accent)" : "var(--warn)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
