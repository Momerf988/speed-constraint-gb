-- Add Primary Keys (The strict uniqueness rules)
ALTER TABLE raw.collision ADD PRIMARY KEY (collision_index);
ALTER TABLE raw.vehicle ADD PRIMARY KEY (collision_index, vehicle_reference);
ALTER TABLE raw.casualty ADD PRIMARY KEY (collision_index, vehicle_reference, casualty_reference);

-- Add Foreign Keys (The strict linking rules)
ALTER TABLE raw.vehicle ADD FOREIGN KEY (collision_index) REFERENCES raw.collision (collision_index);
ALTER TABLE raw.casualty ADD FOREIGN KEY (collision_index, vehicle_reference) REFERENCES raw.vehicle (collision_index, vehicle_reference);