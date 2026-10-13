import ServiceHealthGrid from "@/components/system/ServiceHealthGrid";
import GpuStrip from "@/components/system/GpuStrip";
import ExternalLinks from "@/components/system/ExternalLinks";

export default function System() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">System</h2>
      <ServiceHealthGrid />
      <GpuStrip />
      <ExternalLinks />
    </div>
  );
}
