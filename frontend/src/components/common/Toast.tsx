import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import type { Toast as ToastType } from '../../types';
import './Toast.css';

interface Props {
  toasts: ToastType[];
  onDismiss: (id: number) => void;
}

export function ToastContainer({ toasts, onDismiss }: Props) {
  if (toasts.length === 0) return null;
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onDismiss }: { toast: ToastType; onDismiss: (id: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    gsap.fromTo(el,
      { x: 80, opacity: 0 },
      { x: 0, opacity: 1, duration: 0.35, ease: 'power3.out' }
    );
  }, []);

  const handleDismiss = () => {
    const el = ref.current;
    if (el) {
      gsap.to(el, {
        x: 80, opacity: 0, duration: 0.25, ease: 'power2.in',
        onComplete: () => onDismiss(toast.id),
      });
    } else {
      onDismiss(toast.id);
    }
  };

  const icon = toast.type === 'error' ? 'error' : toast.type === 'success' ? 'check_circle' : 'info';

  return (
    <div ref={ref} className={`toast toast--${toast.type}`} onClick={handleDismiss}>
      <span className="material-icons toast__icon">{icon}</span>
      <span className="toast__message">{toast.message}</span>
    </div>
  );
}
