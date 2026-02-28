/**
 * @jest-environment jsdom
 */

import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MetricCard } from '../MetricCard';

// Mock framer-motion to avoid animation issues in tests
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: { children?: ReactNode;[key: string]: unknown }) => (
      <div {...props}>{children}</div>
    ),
  },
}));

describe('MetricCard Component', () => {
  const defaultProps = {
    title: 'Test Metric',
    value: '$1,000,000',
    subValue: '+2.5%',
    trend: 'up' as const,
    icon: '📈'
  };

  test('renders metric card with basic props', () => {
    render(<MetricCard {...defaultProps} />);

    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('$1,000,000')).toBeInTheDocument();
    expect(screen.getByText('+2.5%')).toBeInTheDocument();
  });

  test('displays positive change correctly', () => {
    render(<MetricCard {...defaultProps} subValue='+5.2%' trend='up' />);

    const changeElement = screen.getByText('+5.2%');
    expect(changeElement).toBeInTheDocument();
  });

  test('displays negative change correctly', () => {
    render(<MetricCard {...defaultProps} subValue='-3.1%' trend='down' />);

    const changeElement = screen.getByText('-3.1%');
    expect(changeElement).toBeInTheDocument();
  });

  test('handles zero change', () => {
    render(<MetricCard {...defaultProps} subValue='0%' trend='neutral' />);

    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  test('displays icon when provided', () => {
    render(<MetricCard {...defaultProps} icon="📊" />);

    expect(screen.getByText('📊')).toBeInTheDocument();
  });

  test('displays subtitle when provided', () => {
    render(<MetricCard {...defaultProps} subValue='Last 30 days' />);

    expect(screen.getByText('Last 30 days')).toBeInTheDocument();
  });

  test('applies custom className', () => {
    const customClass = 'custom-metric-class';
    render(<MetricCard {...defaultProps} className={customClass} />);

    const card = screen.getByText('Test Metric').closest('[class*="glass-card"]');
    expect(card).toBeInTheDocument();
  });

  test('handles large numbers', () => {
    render(<MetricCard {...defaultProps} value="$10,000,000,000" />);

    expect(screen.getByText('$10,000,000,000')).toBeInTheDocument();
  });

  test('handles missing subValue prop', () => {
    const propsWithoutSubValue = {
      title: defaultProps.title,
      value: defaultProps.value,
      icon: defaultProps.icon,
    };

    render(<MetricCard {...propsWithoutSubValue} />);

    expect(screen.queryByText(/\%/)).not.toBeInTheDocument();
  });
});
