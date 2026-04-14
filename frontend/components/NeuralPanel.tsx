"use client";

import React, { useState, useEffect } from 'react';
import { Card, Title, Text, Metric, Flex, ProgressBar, BarList } from '@tremor/react';

export default function NeuralPanel() {
  const [neuralState, setNeuralState] = useState({
    instrument: "BTC/USD",
    state_vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    action: 0,
    confidence: 0.0
  });

  // Mocking the websocket/API stream for the visualization simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setNeuralState({
        instrument: "BTC/USD",
        state_vector: [
          Math.random(), // vPIN
          (Math.random() * 2) - 1, // OBI
          Math.random() * 0.5, // Spread
          Math.random() > 0.8 ? 1.0 : 0.0, // Iceberg
          1.0, // Inventory
          Math.random() // Time Rem
        ],
        action: Math.floor(Math.random() * 4),
        confidence: 0.6 + (Math.random() * 0.4)
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const actionMap = ["Wait (Hold)", "Passive Maker", "Penny-Jump", "Market Sweep"];
  
  const vectorData = [
    { name: 'vPIN Toxicity', value: neuralState.state_vector[0] * 100 },
    { name: 'OBI Pressure', value: Math.abs(neuralState.state_vector[1]) * 100 },
    { name: 'Spread Width', value: neuralState.state_vector[2] * 100 },
    { name: 'Iceberg Flag', value: neuralState.state_vector[3] * 100 },
    { name: 'Inventory %', value: neuralState.state_vector[4] * 100 },
    { name: 'Time Remaining', value: neuralState.state_vector[5] * 100 },
  ];

  return (
    <Card className="mt-4 bg-[#0d1117] ring-1 ring-cyan-500/30">
      <Title className="text-cyan-400">PPO Neural Inference Node</Title>
      <Text className="text-slate-400">6-D L2 Input Vector mapped to Policy Output for {neuralState.instrument}</Text>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <div>
          <Text className="text-white mb-2">Input State Space (Normalized)</Text>
          <BarList data={vectorData} className="mt-2" color="cyan" />
        </div>
        
        <div className="flex flex-col justify-center">
          <Card className="bg-slate-900 border-none ring-1 ring-slate-800">
            <Text className="text-slate-400">Predicted Optimal Action</Text>
            <Metric className="text-white mt-2">{actionMap[neuralState.action]}</Metric>
            
            <Flex className="mt-6">
              <Text className="text-slate-400">Network Confidence</Text>
              <Text className="text-white">{(neuralState.confidence * 100).toFixed(1)}%</Text>
            </Flex>
            <ProgressBar value={neuralState.confidence * 100} color="emerald" className="mt-2" />
          </Card>
        </div>
      </div>
    </Card>
  );
}
