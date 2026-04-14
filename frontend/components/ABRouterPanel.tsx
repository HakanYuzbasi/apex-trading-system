"use client";

import React, { useState, useEffect } from 'react';
import { Card, Title, Text, AreaChart, Metric, Flex, BadgeDelta } from '@tremor/react';

export default function ABRouterPanel() {
  const [chartData, setChartData] = useState<any[]>([]);
  const [summary, setSummary] = useState({
    avgReduction: 0.0,
    cythonCumulativeDrip: 0.0,
    neuralCumulativeDrip: 0.0
  });

  // Mocking A/B telemetry stream
  useEffect(() => {
    let mockData: any[] = [];
    let currentCython = 0;
    let currentNeural = 0;
    
    // Seed initial data
    for (let i = 0; i < 20; i++) {
        currentCython += (Math.random() * 5); // 0-5 bps slip
        currentNeural += (Math.random() * 3); // 0-3 bps slip
        mockData.push({
            Trade: `T-${20-i}`,
            "Cython Slip (bps)": currentCython,
            "Neural Slip (bps)": currentNeural
        });
    }
    
    setChartData(mockData.reverse());
    
    const interval = setInterval(() => {
      setChartData(prev => {
        const last = prev[prev.length - 1];
        const newCython = last["Cython Slip (bps)"] + (Math.random() * 5);
        const newNeural = last["Neural Slip (bps)"] + (Math.random() * 3);
        
        const next = [...prev.slice(1), {
            Trade: `T-Now`,
            "Cython Slip (bps)": newCython,
            "Neural Slip (bps)": newNeural
        }];
        
        setSummary({
            avgReduction: ((newCython - newNeural) / newCython) * 100,
            cythonCumulativeDrip: newCython,
            neuralCumulativeDrip: newNeural
        });
        
        return next;
      });
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="mt-4 bg-[#0d1117] ring-1 ring-fuchsia-500/30">
      <Flex alignItems="start">
        <div>
          <Title className="text-fuchsia-400">A/B Shadow Routing (TCO Analysis)</Title>
          <Text className="text-slate-400">Cython vs Deep RL Slippage Accrual</Text>
        </div>
        <BadgeDelta 
            deltaType={summary.avgReduction > 15 ? "increase" : "moderateIncrease"}
            isIncreasePositive={true}
        >
            {summary.avgReduction.toFixed(2)}% Edge
        </BadgeDelta>
      </Flex>
      
      <div className="grid grid-cols-2 gap-4 mt-6 mb-4">
        <Card className="bg-slate-900 border-none ring-1 ring-slate-800">
            <Text className="text-slate-400">Cython Cumulative Slippage</Text>
            <Metric className="text-rose-500">{summary.cythonCumulativeDrip.toFixed(1)} bps</Metric>
        </Card>
        <Card className="bg-slate-900 border-none ring-1 ring-slate-800">
            <Text className="text-slate-400">Neural Cumulative Slippage</Text>
            <Metric className="text-emerald-500">{summary.neuralCumulativeDrip.toFixed(1)} bps</Metric>
        </Card>
      </div>

      <AreaChart
        className="h-72 mt-4"
        data={chartData}
        index="Trade"
        categories={["Cython Slip (bps)", "Neural Slip (bps)"]}
        colors={["rose", "emerald"]}
        valueFormatter={(number) => `${number.toFixed(1)} bps`}
        showLegend={true}
        yAxisWidth={40}
      />
    </Card>
  );
}
