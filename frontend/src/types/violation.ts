/** Row shape from the Supabase `violations` table. */
export interface Violation {
  id: number;
  license_plate: string;
  evidence_image_path: string | null;
  detected_at: string | null;
  violation_started_at: string | null;
  duration_seconds: number;
  status: string;
  camera_id?: number;
  zone_id?: number;
  telegram_sent?: boolean;
}
