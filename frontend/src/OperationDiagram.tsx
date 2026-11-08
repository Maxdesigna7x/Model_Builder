import type { ReactNode } from "react";

type DiagramProps = {
  type: string;
  properties: Record<string, string | number | boolean>;
};

const style = {
  width: "100%",
  height: "38px",
  display: "block",
  marginTop: "5px",
  color: "var(--accent)",
};

function gridCells({
  columns,
  rows,
  x,
  y,
  size,
  active = () => true,
  accent = () => false,
}: {
  columns: number;
  rows: number;
  x: number;
  y: number;
  size: number;
  active?: (column: number, row: number) => boolean;
  accent?: (column: number, row: number) => boolean;
}) {
  const cells: ReactNode[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      cells.push(
        <rect
          key={`${column}-${row}`}
          x={x + column * size}
          y={y + row * size}
          width={size - 1}
          height={size - 1}
          rx="1"
          fill={accent(column, row) ? "currentColor" : active(column, row) ? "var(--accent-soft)" : "transparent"}
          stroke="currentColor"
          strokeWidth="0.8"
        />,
      );
    }
  }
  return cells;
}

function Arrow() {
  return <path d="M66 19h31m-6-5 6 5-6 5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />;
}

function PoolWindowDiagram({ type, properties }: DiagramProps) {
  const is1d = type.endsWith("1d");
  const isMax = type.startsWith("maxpool");
  const rawKernel = Number(properties.kernel_size ?? 2);
  const kernel = Number.isFinite(rawKernel) ? Math.max(2, Math.min(4, Math.round(rawKernel))) : 2;
  const showExample = kernel === 2;

  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {is1d ? (
        <>
          {Array.from({ length: kernel }, (_, index) => (
            <g key={index}>
              <rect x={8 + index * (42 / kernel)} y="12" width={42 / kernel - 1} height="15" rx="1" fill={isMax && showExample && index === 1 ? "currentColor" : "var(--accent-soft)"} stroke="currentColor" strokeWidth="0.8" />
              {showExample && <text x={8 + (index + 0.5) * (42 / kernel)} y="22" textAnchor="middle" fill={isMax && index === 1 ? "var(--surface)" : "currentColor"} fontSize="12">{index === 0 ? 1 : 4}</text>}
            </g>
          ))}
          <Arrow />
          <rect x="132" y="11" width="19" height="17" rx="2" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="1" />
          {showExample && <text x="141.5" y="22" textAnchor="middle" fill="currentColor" fontSize="12">{isMax ? 4 : "2.5"}</text>}
        </>
      ) : (
        <>
          {gridCells({
            columns: kernel,
            rows: kernel,
            x: 8,
            y: 3,
            size: 27 / kernel,
            accent: (column, row) => isMax && showExample && column === 1 && row === 1,
          })}
          {showExample && [
            [1, 2],
            [3, 4],
          ].map((row, rowIndex) => row.map((value, columnIndex) => (
            <text
              key={`${columnIndex}-${rowIndex}`}
              x={8 + (columnIndex + 0.5) * (27 / kernel) - 0.5}
              y={3 + (rowIndex + 0.68) * (27 / kernel)}
              textAnchor="middle"
              fill={isMax && columnIndex === 1 && rowIndex === 1 ? "var(--surface)" : "currentColor"}
              fontSize="12"
            >{value}</text>
          )))}
          <Arrow />
          <rect x="132" y="10" width="18" height="18" rx="2" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="1" />
          {showExample && <text x="141" y="22" textAnchor="middle" fill="currentColor" fontSize="12">{isMax ? 4 : "2.5"}</text>}
        </>
      )}
    </svg>
  );
}

function GlobalAverageDiagram({ type, properties }: DiagramProps) {
  const is1d = type.endsWith("1d");
  const rawSize = Number(properties.output_size ?? 1);
  const outputSize = Number.isFinite(rawSize) ? Math.max(1, Math.min(4, Math.round(rawSize))) : 1;

  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {is1d ? (
        <>
          {Array.from({ length: 8 }, (_, index) => (
            <rect key={index} x={8 + index * 5} y="11" width="4" height="16" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.7" />
          ))}
          <Arrow />
          {Array.from({ length: outputSize }, (_, index) => (
            <rect key={index} x={132 + index * (18 / outputSize)} y="13" width={18 / outputSize - 1} height="12" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.8" />
          ))}
        </>
      ) : (
        <>
          {gridCells({ columns: 4, rows: 4, x: 8, y: 3, size: 7, accent: (column, row) => (column + row) % 2 === 0 })}
          <Arrow />
          {gridCells({ columns: outputSize, rows: outputSize, x: 132, y: 3, size: 27 / outputSize })}
        </>
      )}
    </svg>
  );
}

function ShapeDiagram({ type }: DiagramProps) {
  if (type === "permute") {
    return (
      <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
        {gridCells({ columns: 3, rows: 2, x: 12, y: 7, size: 11 })}
        <Arrow />
        {gridCells({ columns: 2, rows: 3, x: 137, y: 2, size: 11 })}
      </svg>
    );
  }

  const flatten = type === "flatten";
  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {flatten ? gridCells({ columns: 2, rows: 2, x: 12, y: 5, size: 13 }) : Array.from({ length: 4 }, (_, index) => (
        <rect key={index} x={12 + index * 8} y="14" width="7" height="11" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.8" />
      ))}
      <Arrow />
      {flatten ? Array.from({ length: 4 }, (_, index) => (
        <rect key={index} x={133 + index * 8} y="14" width="7" height="11" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.8" />
      )) : gridCells({ columns: 2, rows: 2, x: 137, y: 5, size: 13 })}
    </svg>
  );
}

function MergeDiagram({ type }: DiagramProps) {
  if (type === "add") {
    return (
      <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
        {gridCells({ columns: 2, rows: 2, x: 4, y: 5, size: 12 })}
        {gridCells({ columns: 2, rows: 2, x: 40, y: 5, size: 12 })}
        <text x="69" y="22" textAnchor="middle" fill="currentColor" fontSize="15">+</text>
        <path d="M79 19h17m-5-4 5 4-5 4" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        {gridCells({ columns: 2, rows: 2, x: 129, y: 5, size: 12, accent: () => true })}
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      <rect x="7" y="8" width="30" height="8" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.8" />
      <rect x="7" y="21" width="30" height="8" rx="1" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="0.8" />
      <path d="M48 13h12v6h10m-10 0v8h10m-4-4 4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="91" y="11" width="59" height="16" rx="2" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="1" />
      <path d="M120 11v16" stroke="currentColor" strokeWidth="0.8" strokeDasharray="2 2" />
    </svg>
  );
}

function UpsampleDiagram() {
  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {gridCells({ columns: 2, rows: 2, x: 9, y: 5, size: 13 })}
      <Arrow />
      {gridCells({ columns: 4, rows: 4, x: 123, y: 1, size: 7, accent: (column, row) => (column + row) % 2 === 0 })}
    </svg>
  );
}

function SequenceDiagram({ type, properties }: DiagramProps) {
  const mode = String(properties.mode ?? (type === "temporal_select" ? "last" : "mean")).toLowerCase();
  const selectedIndex = mode === "cls" || mode === "first" ? 0 : 4;
  const selectOne = type === "temporal_select" || mode === "last" || mode === "cls" || mode === "first";
  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {Array.from({ length: 5 }, (_, index) => (
        <rect key={index} x={8 + index * 9} y="12" width="8" height="14" rx="1" fill={selectOne && index === selectedIndex ? "currentColor" : "var(--accent-soft)"} stroke="currentColor" strokeWidth="0.8" />
      ))}
      <Arrow />
      <rect x="132" y="10" width="19" height="18" rx="2" fill="var(--accent-soft)" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}

function DropoutDiagram({ type }: DiagramProps) {
  const is2d = type === "dropout2d";
  return (
    <svg viewBox="0 0 164 38" style={style} aria-hidden="true" focusable="false">
      {gridCells({ columns: 4, rows: 3, x: 13, y: 4, size: 8, accent: (column, row) => is2d ? column === 2 : (column === 1 && row === 1) || (column === 3 && row === 0) })}
      <Arrow />
      {gridCells({ columns: 4, rows: 3, x: 123, y: 4, size: 8, active: (column, row) => is2d ? column !== 2 : !((column === 1 && row === 1) || (column === 3 && row === 0)) })}
    </svg>
  );
}

export default function OperationDiagram({ type, properties }: DiagramProps) {
  if (/^(maxpool|avgpool)(1d|2d)$/.test(type)) return <PoolWindowDiagram type={type} properties={properties} />;
  if (/^adaptiveavgpool(1d|2d)$/.test(type)) return <GlobalAverageDiagram type={type} properties={properties} />;
  if (["flatten", "reshape", "permute"].includes(type)) return <ShapeDiagram type={type} properties={properties} />;
  if (type === "add" || type === "concat") return <MergeDiagram type={type} properties={properties} />;
  if (type === "resize2d" || type === "convtranspose2d") return <UpsampleDiagram />;
  if (type === "temporal_select" || type === "sequence_pool") return <SequenceDiagram type={type} properties={properties} />;
  if (type === "dropout" || type === "dropout2d") return <DropoutDiagram type={type} properties={properties} />;
  return null;
}
