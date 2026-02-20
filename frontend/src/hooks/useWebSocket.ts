"use client";
import { useRef, useState, useCallback, useEffect } from "react";
import type { ServerMessage } from "@/lib/types";

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    console.log("[WS] connect called, current state:", wsRef.current?.readyState);

    // Close any existing connection first
    if (wsRef.current) {
      console.log("[WS] Closing existing connection, readyState:", wsRef.current.readyState);
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    console.log("[WS] Creating new WebSocket to", url);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current === ws) {
        console.log("[WS] Connected");
        setIsConnected(true);
        setError(null);
      } else {
        console.log("[WS] onopen fired on stale socket, ignoring");
      }
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return;
      try {
        const data = JSON.parse(event.data) as ServerMessage;
        setLastMessage(data);
      } catch {
        // ignore non-JSON messages
      }
    };

    ws.onclose = (event) => {
      console.log("[WS] onclose, code:", event.code, "reason:", event.reason, "isCurrent:", wsRef.current === ws);
      if (wsRef.current === ws) {
        setIsConnected(false);
      }
    };

    ws.onerror = (event) => {
      console.error("[WS] onerror", event);
      if (wsRef.current === ws) {
        setError("WebSocket connection failed");
        setIsConnected(false);
      }
    };
  }, [url]);

  const disconnect = useCallback(() => {
    console.log("[WS] disconnect called, hasSocket:", !!wsRef.current, "readyState:", wsRef.current?.readyState);
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendFrame = useCallback((blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(blob);
    }
  }, []);

  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return { isConnected, lastMessage, error, connect, disconnect, sendFrame, sendCommand };
}
