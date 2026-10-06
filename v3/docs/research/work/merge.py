import csv,glob,re,sys
files=sorted(glob.glob("t[1-57]_related.csv"))
rows={}; bydoi={}; dup=[]
for f in files:
    for r in csv.DictReader(open(f)):
        k=r["key"].strip(); d=r["doi"].strip().lower()
        if d and d in bydoi and bydoi[d]!=k: dup.append((k,bydoi[d],d)); k=bydoi[d]
        if k in rows:
            o=rows[k]
            o["topic"]=";".join(sorted(set(o["topic"].split(";"))|set(r["topic"].split(";"))))
            o["closeness_1to5"]=str(max(int(o["closeness_1to5"] or 0),int(r["closeness_1to5"] or 0)))
            if f.startswith("t7"):  # threat read supersedes earlier reads
                for c in ("what_it_does","not_found_in_sections_read","sections_read","notes","verified","verify_method"): 
                    if r[c]: o[c]=r[c]
            else:
                o["notes"]=(o["notes"]+" | "+r["notes"]).strip(" |")
        else:
            rows[k]=dict(r)
        if d: bydoi[d]=k
w=csv.DictWriter(open("merged_related.csv","w",newline=""),fieldnames=list(csv.DictReader(open(files[0])).fieldnames))
w.writeheader()
for k in sorted(rows,key=lambda k:(-int(rows[k]["closeness_1to5"] or 0),k)): w.writerow(rows[k])
print(len(rows),"unique; doi-dups merged:",dup)
