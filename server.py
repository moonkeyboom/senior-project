from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import uvicorn
import traceback

# Import the existing CPD logic
from optimal_cpd_omega_prime import dp_best, exhaustive_best_for_k, assign_labels, describe

app = FastAPI(title="CPD Calculator API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/calculate")
async def calculate_cpd(
    file: UploadFile = File(...),
    labels: str = Form(...)
):
    try:
        # Read file
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or XLSX.")
        
        # Find score column
        col = None
        for c in df.columns:
            if str(c).lower() in ("score", "scores", "performance", "value"):
                col = c
                break
        if col is None:
            num = df.select_dtypes(include=[np.number])
            if num.empty:
                raise HTTPException(status_code=400, detail="No numeric columns found in data.")
            col = num.columns[0]
            
        vals = df[col].dropna().to_numpy(dtype=float)
        
        # Parse labels
        grade_symbols = [g.strip() for g in labels.split(",") if g.strip()]
        if not grade_symbols:
            raise HTTPException(status_code=400, detail="Labels cannot be empty.")
            
        num_labels = len(grade_symbols)
        U, L = 100.0, 0.0 # Default bounds
        
        v = np.sort(vals)[::-1]
        kL = min(num_labels, len(v))
        
        # Run all available methods
        methods_evaluated = []
        
        # 1. DP method
        res_dp = dp_best(v, num_labels, U, L, k=kL)
        if res_dp:
            methods_evaluated.append({
                "name": "1-D k-means (Dynamic Programming)",
                "res": res_dp
            })
            
        # 2. Exhaustive Search
        # (Warning: could be slow for large N, but required by user logic)
        res_ex = exhaustive_best_for_k(v, kL, num_labels, U, L)
        if res_ex:
            methods_evaluated.append({
                "name": "Exhaustive Search (Combinatorial)",
                "res": res_ex
            })
            
        if not methods_evaluated:
            raise HTTPException(status_code=500, detail="Failed to calculate CPD partitions with any method.")
            
        # Find optimal method (highest omega_prime)
        methods_evaluated.sort(key=lambda x: x["res"].omega_prime, reverse=True)
        best_match = methods_evaluated[0]
        res = best_match["res"]
        method_name = best_match["name"]
        
        # Format scores to send back
        method_scores = [
            {"name": m["name"], "omega_prime": round(m["res"].omega_prime, 4)}
            for m in methods_evaluated
        ]
            
        labeled, syms = assign_labels(v, res.cuts, grade_symbols)
        
        clusters = []
        for i, s in enumerate(syms):
            vals_in_cluster = [x[0] for x in labeled if x[1] == s]
            if vals_in_cluster:
                clusters.append({
                    "grade": s,
                    "min": float(min(vals_in_cluster)),
                    "max": float(max(vals_in_cluster)),
                    "amount": len(vals_in_cluster)
                })
                
        labeled_records = [{"score": float(x[0]), "label": x[1]} for x in labeled]
            
        return {
            "omega_prime": round(res.omega_prime, 4),
            "omega1": round(res.omega1, 4),
            "omega2": round(res.omega2, 4),
            "omega3": round(res.omega3, 4),
            "sigma": round(res.sigma, 4),
            "clusters": clusters,
            "labeled_records": labeled_records,
            "n": len(vals),
            "method_name": method_name,
            "method_scores": method_scores
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
