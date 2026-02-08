"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { DollarSign } from "lucide-react";

interface SectorChartProps {
    data: Record<string, number>;
}

const COLORS = [
    '#00f3ff', // Cyan
    '#7000ff', // Purple
    '#ff0055', // Red/Pink
    '#ffd700', // Gold
    '#00ff9d', // Green
    '#ffaa00', // Orange
    '#00aaff', // Blue
];

export default function SectorChart({ data }: SectorChartProps) {
    const chartData = Object.entries(data).map(([name, value]) => ({
        name,
        value: value
    })).filter(item => item.value > 0);

    // Sort by value desc
    chartData.sort((a, b) => b.value - a.value);

    return (
        <div className="glass-panel p-6 rounded-xl relative h-full flex flex-col">
            <div className="flex justify-between items-start mb-4">
                <span className="text-muted-foreground text-sm font-medium uppercase tracking-wider">Sector Exposure</span>
                <DollarSign className="w-4 h-4 text-muted-foreground" />
            </div>

            <div className="flex-1 min-h-[250px] relative">
                {chartData.length === 0 ? (
                    <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
                        No Exposure Data
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(10, 10, 15, 0.9)',
                                    borderColor: 'rgba(255,255,255,0.1)',
                                    borderRadius: '8px',
                                    color: '#fff'
                                }}
                                formatter={(value: number | undefined) => `${((value || 0) * 100).toFixed(1)}%`}
                            />
                            <Legend
                                verticalAlign="bottom"
                                height={36}
                                formatter={(value) => <span style={{ color: '#ccc', fontSize: '12px' }}>{value}</span>}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Center Label for Total Allocation */}
            <div className="absolute top-[55%] left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none text-center">
                <div className="text-xs text-muted-foreground">Alloc</div>
                <div className="text-sm font-bold text-white">
                    {(chartData.reduce((acc, item) => acc + item.value, 0) * 100).toFixed(0)}%
                </div>
            </div>
        </div>
    );
}
