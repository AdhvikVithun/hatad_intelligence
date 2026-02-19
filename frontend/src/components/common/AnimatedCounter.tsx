import { useEffect, useRef } from 'react';
import gsap from 'gsap';

interface Props {
  value: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  className?: string;
  style?: React.CSSProperties;
}

export function AnimatedCounter({
  value, duration = 1.2, suffix = '', prefix = '',
  decimals = 0, className = '', style,
}: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const prev = useRef(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const counter = { val: prev.current };
    const tween = gsap.to(counter, {
      val: value,
      duration,
      ease: 'power2.out',
      onUpdate: () => {
        el.textContent = `${prefix}${counter.val.toFixed(decimals)}${suffix}`;
      },
    });

    prev.current = value;
    return () => { tween.kill(); };
  }, [value, duration, suffix, prefix, decimals]);

  return (
    <span ref={ref} className={className} style={style}>
      {prefix}{value.toFixed(decimals)}{suffix}
    </span>
  );
}
