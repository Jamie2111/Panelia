"use client";

/**
 * transport-icons.tsx
 *
 * Self-contained SVG icons for the JKL transport row. Designed to read at
 * 16-20px without losing meaning. All three share the same triangle/bar
 * vocabulary so the row scans as a related family.
 */

import * as React from "react";

interface IconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
}

function Svg({ size = 16, children, ...rest }: React.PropsWithChildren<IconProps>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      stroke="none"
      aria-hidden
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconRewind = (props: IconProps) => (
  <Svg {...props}>
    {/* Two stacked left-pointing triangles (familiar rewind glyph). */}
    <path d="M11 6 5 12l6 6V6z" />
    <path d="M19 6l-6 6 6 6V6z" />
  </Svg>
);

export const IconPause = (props: IconProps) => (
  <Svg {...props}>
    <rect x="6" y="5" width="4" height="14" rx="1" />
    <rect x="14" y="5" width="4" height="14" rx="1" />
  </Svg>
);

export const IconPlay = (props: IconProps) => (
  <Svg {...props}>
    {/* Right-pointing triangle */}
    <path d="M8 5v14l11-7L8 5z" />
  </Svg>
);

export const IconZoomIn = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="11" cy="11" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
    <path d="M11 8v6M8 11h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    <path d="m20 20-3-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </Svg>
);

export const IconZoomOut = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="11" cy="11" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
    <path d="M8 11h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    <path d="m20 20-3-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </Svg>
);

export const IconMarkIn = (props: IconProps) => (
  <Svg {...props}>
    <path d="M6 4v16h2V4H6zm5 0v16l9-8-9-8z" />
  </Svg>
);

export const IconMarkOut = (props: IconProps) => (
  <Svg {...props}>
    <path d="M18 4v16h-2V4h2zm-5 0v16l-9-8 9-8z" />
  </Svg>
);

export const IconSave = (props: IconProps) => (
  <Svg {...props}>
    <path d="M5 3h11l3 3v15H5V3zm2 2v6h8V5H7zm0 8v6h10v-6H7z" />
  </Svg>
);
