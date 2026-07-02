-- Run this in Supabase Dashboard -> SQL Editor

create table if not exists public.violations (
  id bigint generated always as identity primary key,
  vehicle_id bigint,
  camera_id integer not null default 1,
  zone_id integer not null default 1,
  license_plate text not null,
  detected_at timestamptz not null default now(),
  duration_seconds integer not null default 0,
  evidence_image_path text not null,
  status text not null default 'Pending',
  telegram_sent boolean not null default false
);

-- Storage bucket for violation photos (create in Dashboard if this insert fails)
insert into storage.buckets (id, name, public)
values ('violation-images', 'violation-images', true)
on conflict (id) do nothing;
