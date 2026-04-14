import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    // The run_state folder is located at the root of the project: apex-trading/run_state
    const archivePath = path.resolve(process.cwd(), "../run_state/v3/briefings_archive.json");
    
    if (!fs.existsSync(archivePath)) {
      return NextResponse.json({ briefings: [] }, { status: 200 });
    }
    
    const fileData = fs.readFileSync(archivePath, "utf-8");
    const briefings = JSON.parse(fileData);
    
    // Sort descending by date
    briefings.sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    return NextResponse.json({ briefings }, { status: 200 });
  } catch (error) {
    console.error("Error reading briefings archive:", error);
    return NextResponse.json({ error: "Failed to load briefings archive" }, { status: 500 });
  }
}
