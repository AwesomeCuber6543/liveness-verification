"use client";
import { useState } from "react";
import ActiveVerification from "@/components/ActiveVerification";

export default function ActiveVerificationPage() {
  const [sessionKey] = useState(() => Date.now());
  return <ActiveVerification key={sessionKey} />;
}
