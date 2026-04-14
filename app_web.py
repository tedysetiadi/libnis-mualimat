
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename

import engine_web as engine

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
GRAPH_FOLDER = os.path.join("static", "graphs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)


def draw_graph_to_file(G, filepath, title, max_nodes=70, max_labels=30):
    if G.number_of_nodes() == 0:
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Graf kosong", ha="center", va="center", fontsize=16)
        plt.title(title)
        plt.axis("off")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return

    if G.number_of_nodes() > max_nodes:
        wd = dict(G.degree(weight="weight"))
        keep = sorted(wd, key=wd.get, reverse=True)[:max_nodes]
        G_plot = G.subgraph(keep).copy()
    else:
        G_plot = G.copy()

    fig = plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(G_plot, seed=42, k=0.7)
    weights = [G_plot[u][v].get("weight", 1) for u, v in G_plot.edges()]
    widths = [0.4 + min(w, 5) * 0.35 for w in weights]

    nx.draw_networkx_nodes(G_plot, pos, node_size=90, alpha=0.85)
    nx.draw_networkx_edges(G_plot, pos, width=widths, alpha=0.25)

    if G_plot.number_of_nodes() <= max_labels:
        labels = {n: G_plot.nodes[n].get("label", n) for n in G_plot.nodes()}
        nx.draw_networkx_labels(G_plot, pos, labels=labels, font_size=7)

    plt.title(f"{title}\nNode={G_plot.number_of_nodes()} | Edge={G_plot.number_of_edges()}")
    plt.axis("off")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="Silakan upload file Excel terlebih dahulu.")

        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        saved_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")
        file.save(saved_path)

        try:
            df_raw = engine.load_data(saved_path)

            start_date = request.form.get("start_date") or None
            end_date = request.form.get("end_date") or None
            kelompok = request.form.get("kelompok") or None
            status = request.form.get("status") or None

            df = engine.filter_data(
                df_raw,
                start_date=start_date,
                end_date=end_date,
                kelompok=kelompok,
                status=status
            )

            summary = engine.circulation_summary(df)
            G_books = engine.build_book_graph(df)
            G_members = engine.build_member_graph(df)

            m_books = engine.compute_metrics(G_books)
            m_members = engine.compute_metrics(G_members)

            kelompok_df = engine.kelompok_stats(df)
            monthly_df = engine.monthly_stats(df)
            top_titles_df = engine.top_titles(df, 10)
            top_members_df = engine.top_members(df, 10)
            communities = engine.detect_communities(G_books)

            book_graph_file = f"book_graph_{unique_id}.png"
            member_graph_file = f"member_graph_{unique_id}.png"

            draw_graph_to_file(G_books, os.path.join(GRAPH_FOLDER, book_graph_file), "Graf Buku–Buku")
            draw_graph_to_file(G_members, os.path.join(GRAPH_FOLDER, member_graph_file), "Graf Anggota–Anggota")

            excel_path = os.path.join(OUTPUT_FOLDER, f"hasil_analisis_{unique_id}.xlsx")
            engine.export_excel(excel_path, {
                "DataTerfilter": df,
                "RankingBuku": m_books,
                "RankingAnggota": m_members,
                "KelompokAktif": kelompok_df,
                "TrenBulanan": monthly_df,
                "TopJudul": top_titles_df,
                "TopAnggota": top_members_df
            })

            gexf_books = os.path.join(OUTPUT_FOLDER, f"graf_buku_{unique_id}.gexf")
            gexf_members = os.path.join(OUTPUT_FOLDER, f"graf_anggota_{unique_id}.gexf")
            engine.export_gexf(G_books, gexf_books)
            engine.export_gexf(G_members, gexf_members)

            return render_template(
                "index.html",
                success="Analisis berhasil.",
                summary=summary,
                book_table=m_books.head(15).to_html(classes="table table-striped", index=False),
                member_table=m_members.head(15).to_html(classes="table table-striped", index=False),
                kelompok_table=kelompok_df.to_html(classes="table table-striped", index=False),
                monthly_table=monthly_df.to_html(classes="table table-striped", index=False),
                top_titles_table=top_titles_df.to_html(classes="table table-striped", index=False),
                top_members_table=top_members_df.to_html(classes="table table-striped", index=False),
                communities=communities,
                book_graph=book_graph_file,
                member_graph=member_graph_file,
                excel_file=os.path.basename(excel_path),
                gexf_books=os.path.basename(gexf_books),
                gexf_members=os.path.basename(gexf_members),
            )

        except Exception as e:
            return render_template("index.html", error=f"Gagal memproses data: {e}")

    return render_template("index.html")


@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("index"))


#if __name__ == "__main__":
 #   app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)