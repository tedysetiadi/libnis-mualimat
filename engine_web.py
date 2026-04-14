#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import combinations

import pandas as pd
import networkx as nx


def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _norm_col(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    for ch in [".", ":", "-", "_", "/", "\\"]:
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def infer_kelompok_anggota(member_id):
    s = clean_text(member_id)
    if s == "":
        return "Tidak diketahui"

    s_up = s.upper()
    if s_up.startswith("MA"):
        return "MA"
    if s_up.startswith("MTS") or s_up.startswith("MTS."):
        return "MTs"
    if s.isdigit():
        return "ID Numerik"
    return "Lainnya"


def ensure_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def load_data(file_path_or_buffer):
    required_aliases = {
        "MemberID": ["ID Anggota", "Id Anggota", "No Anggota", "No. Anggota", "Nomor Anggota", "Member ID"],
        "MemberName": ["Nama Anggota", "Nama", "Nama Peminjam", "Peminjam", "Nama Member"],
        "CopyCode": ["Kode Eksemplar", "Kode Buku", "Barcode", "No Inventaris", "No. Inventaris"],
        "Title": ["Judul", "Judul Buku", "Nama Buku", "Title"],
        "BorrowDate": ["Tanggal Pinjam", "Tgl Pinjam", "Borrow Date", "Tanggal Peminjaman"],
        "ReturnDate": ["Tanggal Kembali", "Tgl Kembali", "Return Date"],
        "Status": ["Status peminjaman", "Status", "Loan Status"]
    }

    raw = pd.read_excel(file_path_or_buffer, header=0)
    norm_to_real = {_norm_col(c): c for c in raw.columns}

    def pick_col(canonical):
        for alias in required_aliases[canonical]:
            key = _norm_col(alias)
            if key in norm_to_real:
                return norm_to_real[key]
        return None

    chosen = {k: pick_col(k) for k in required_aliases}
    missing = [k for k, v in chosen.items() if v is None and k in ["MemberID", "MemberName", "CopyCode", "Title", "BorrowDate"]]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. Kolom terbaca: {list(raw.columns)}")

    use_cols = [v for v in chosen.values() if v is not None]
    df = raw[use_cols].copy()
    rename_map = {v: k for k, v in chosen.items() if v is not None}
    df.rename(columns=rename_map, inplace=True)

    for c in ["ReturnDate", "Status"]:
        if c not in df.columns:
            df[c] = None

    for c in ["MemberID", "MemberName", "CopyCode", "Title", "Status"]:
        df[c] = df[c].apply(clean_text)

    df["BorrowDate"] = ensure_datetime(df["BorrowDate"])
    df["ReturnDate"] = ensure_datetime(df["ReturnDate"])

    df = df[(df["MemberID"] != "") & (df["Title"] != "")].copy()

    df["Status"] = df["Status"].replace({"0": "Kembali", "1": "Dipinjam"})
    df["Status"] = df["Status"].fillna("").replace("", "Tidak diketahui")

    df["KelompokAnggota"] = df["MemberID"].apply(infer_kelompok_anggota)
    df["BorrowMonth"] = df["BorrowDate"].dt.to_period("M").astype(str)

    # pakai judul untuk graf agar pola minat baca muncul
    df["book_node"] = "B_" + df["Title"].str.lower().str.strip()
    df["member_node"] = "M_" + df["MemberID"].astype(str)

    return df


def filter_data(df, start_date=None, end_date=None, kelompok=None, status=None):
    out = df.copy()

    if start_date:
        out = out[out["BorrowDate"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["BorrowDate"] <= pd.to_datetime(end_date)]
    if kelompok and kelompok != "Semua":
        out = out[out["KelompokAnggota"] == kelompok]
    if status and status != "Semua":
        out = out[out["Status"] == status]

    return out


def build_book_graph(df):
    G = nx.Graph()
    for book_node, title in df[["book_node", "Title"]].drop_duplicates().itertuples(index=False):
        G.add_node(book_node, label=title, kind="Buku")

    grouped = df.groupby("member_node")["book_node"].apply(lambda x: list(dict.fromkeys(x)))
    for books in grouped:
        for a, b in combinations(books, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


def build_member_graph(df):
    G = nx.Graph()
    for member_node, member_id, member_name, kelompok in df[["member_node", "MemberID", "MemberName", "KelompokAnggota"]].drop_duplicates().itertuples(index=False):
        G.add_node(member_node, label=member_name, kind="Anggota", member_id=member_id, kelompok=kelompok)

    grouped = df.groupby("book_node")["member_node"].apply(lambda x: list(dict.fromkeys(x)))
    for members in grouped:
        for a, b in combinations(members, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


def compute_metrics(G):
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["Node", "Label", "Degree", "WeightedDegree", "Betweenness", "Closeness", "CBCI"])

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, normalized=True, weight=None)
    closeness = nx.closeness_centrality(G)

    max_deg = max(degree.values()) if degree else 1
    max_bet = max(betweenness.values()) if betweenness else 1

    rows = []
    for node in G.nodes():
        deg = degree.get(node, 0)
        wdeg = weighted_degree.get(node, 0)
        bet = betweenness.get(node, 0.0)
        clo = closeness.get(node, 0.0)
        cbci = (deg / max_deg if max_deg else 0) + (bet / max_bet if max_bet else 0)

        rows.append({
            "Node": node,
            "Label": G.nodes[node].get("label", node),
            "Degree": int(deg),
            "WeightedDegree": float(wdeg),
            "Betweenness": float(bet),
            "Closeness": float(clo),
            "CBCI": float(cbci),
        })

    return pd.DataFrame(rows).sort_values(["CBCI", "WeightedDegree"], ascending=False)


def detect_isolated(G):
    return [G.nodes[n].get("label", n) for n in G.nodes() if G.degree(n) == 0]


def detect_communities(G):
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return []

    communities = nx.community.greedy_modularity_communities(G, weight="weight")
    result = []
    for idx, comm in enumerate(communities, start=1):
        labels = [G.nodes[n].get("label", n) for n in comm]
        result.append({
            "Community": idx,
            "JumlahNode": len(comm),
            "ContohAnggota": ", ".join(labels[:8]) + (" ..." if len(labels) > 8 else "")
        })
    return result


def kelompok_stats(df):
    return (
        df.groupby("KelompokAnggota")
        .size()
        .reset_index(name="JumlahTransaksi")
        .sort_values("JumlahTransaksi", ascending=False)
    )


def monthly_stats(df):
    return (
        df.dropna(subset=["BorrowDate"])
        .groupby("BorrowMonth")
        .size()
        .reset_index(name="JumlahPeminjaman")
        .sort_values("BorrowMonth")
    )


def top_titles(df, n=10):
    return (
        df.groupby("Title")
        .size()
        .reset_index(name="JumlahPeminjaman")
        .sort_values("JumlahPeminjaman", ascending=False)
        .head(n)
    )


def top_members(df, n=10):
    return (
        df.groupby(["MemberID", "MemberName", "KelompokAnggota"])
        .size()
        .reset_index(name="JumlahPeminjaman")
        .sort_values("JumlahPeminjaman", ascending=False)
        .head(n)
    )


def circulation_summary(df):
    return {
        "total_transaksi": int(len(df)),
        "total_judul": int(df["Title"].nunique()),
        "total_anggota": int(df["MemberID"].nunique()),
    }


def export_excel(output_path, sheets):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def export_gexf(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_gexf(G, path)