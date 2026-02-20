"use client";
import { useState } from "react";
import PassiveVerification from "@/components/PassiveVerification";

// This wrapper forces a full remount of the verification component
// every time the page is visited by using a key that changes.
export default function PassiveVerificationPage() {
  const [sessionKey] = useState(() => Date.now());
  return <PassiveVerification key={sessionKey} />;
}
