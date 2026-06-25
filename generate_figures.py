######################################################################
# FILE START: generate_figures.py   (125 lines)
######################################################################
import json, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9.5, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'lines.linewidth': 2.0, 'lines.markersize': 7,
})
C_PRIV='#1a6faf'; C_PT='#2ca02c'; C_SDTC='#d62728'; C_GRAY='#7f7f7f'
os.makedirs('figures', exist_ok=True)

def fig1():
    with open('results/exp1_accuracy.json') as f: data=json.load(f)
    r=data['results']
    keys=['plaintext','privpathinfer','sdtc_5bins','sdtc_10bins','sdtc_20bins','sdtc_50bins','sdtc_100bins']
    labels=['Plaintext DT','PrivPathInfer','SDTC 5B','SDTC 10B','SDTC 20B','SDTC 50B','SDTC 100B']
    colors=[C_PT,C_PRIV,C_SDTC,'#ff7f0e','#9467bd','#8c564b','#e377c2']
    means=[r[k]['mean']*100 for k in keys]; stds=[r[k]['std']*100 for k in keys]
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(len(labels))
    for i in range(len(labels)):
        ax.bar(x[i],means[i],yerr=stds[i],capsize=4,color=colors[i],alpha=0.85,edgecolor='white',linewidth=0.8)
        ax.text(x[i],means[i]+stds[i]+0.3,f'{means[i]:.1f}%',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
    pt=r['plaintext']['mean']*100
    ax.axhline(pt,color=C_PT,linestyle='--',alpha=0.7,linewidth=1.5)
    ax.text(6.4,pt+0.3,f'Plaintext {pt:.1f}%',fontsize=8,color=C_PT,ha='right')
    ax.annotate('0% accuracy loss',xy=(1,means[1]),xytext=(2,79),fontsize=8.5,color=C_PRIV,arrowprops=dict(arrowstyle='->',color=C_PRIV))
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=15,ha='right',fontsize=9)
    ax.set_ylabel('Classification Accuracy (%)'); ax.set_ylim(55,82)
    ax.set_title('Figure 1: Accuracy Comparison — PrivPathInfer vs SDTC\n(PIMA Diabetes, 5-fold CV, max_depth=5)',pad=10)
    plt.tight_layout(); plt.savefig('figures/fig1_accuracy.pdf'); plt.savefig('figures/fig1_accuracy.png'); plt.close()
    print("[DONE] Figure 1")

def fig2():
    with open('results/exp2_storage.json') as f: data=json.load(f)
    r=data['results']; depths=r['depths']; priv=r['privpathinfer_paths']; sdtc=r['sdtc_entries']
    fig,ax=plt.subplots(figsize=(7,5))
    ax.semilogy(depths,sdtc,'o-',color=C_SDTC,linewidth=2,markersize=7,label='SDTC: O(2^depth) entries')
    ax.semilogy(depths,priv,'s-',color=C_PRIV,linewidth=2,markersize=7,label='PrivPathInfer: O(N) paths')
    ax.fill_between(depths,priv,sdtc,alpha=0.08,color=C_PRIV)
    ax.annotate('39.8× fewer\npaths at depth 12',xy=(12,priv[-1]),xytext=(8.5,500),fontsize=9,color=C_PRIV,arrowprops=dict(arrowstyle='->',color=C_PRIV))
    ax.set_xlabel('Tree Depth'); ax.set_ylabel('Entries / Paths (log scale)')
    ax.set_title('Figure 2: Storage — Entry/Path Count\n(PrivPathInfer O(N) vs SDTC O(2^N))',pad=10)
    ax.set_xticks(depths); ax.legend(); ax.set_xlim(1.5,12.5)
    plt.tight_layout(); plt.savefig('figures/fig2_storage_count.pdf'); plt.savefig('figures/fig2_storage_count.png'); plt.close()
    print("[DONE] Figure 2")

def fig3():
    with open('results/exp2_storage.json') as f: data=json.load(f)
    r=data['results']; depths=r['depths']; priv=r['privpathinfer_bytes_kb']; sdtc=r['sdtc_bytes_kb']
    fig,ax=plt.subplots(figsize=(7,5))
    ax.semilogy(depths,sdtc,'o-',color=C_SDTC,linewidth=2,markersize=7,label='SDTC: 32B × 2^depth (AES-128)')
    ax.semilogy(depths,priv,'s-',color=C_PRIV,linewidth=2,markersize=7,label='PrivPathInfer: 290B × N paths (Paillier)')
    ax.text(4,8,'Paillier 256B/rule\nvs AES 16B/entry\n(IND-CPA security cost)',fontsize=8.5,color=C_GRAY,bbox=dict(boxstyle='round',facecolor='white',edgecolor=C_GRAY,alpha=0.8))
    ax.set_xlabel('Tree Depth'); ax.set_ylabel('Encrypted Storage (KB, log scale)')
    ax.set_title('Figure 3: Storage — Encrypted Bytes\n(Paillier IND-CPA overhead vs AES)',pad=10)
    ax.set_xticks(depths); ax.legend(); ax.set_xlim(1.5,12.5)
    plt.tight_layout(); plt.savefig('figures/fig3_storage_bytes.pdf'); plt.savefig('figures/fig3_storage_bytes.png'); plt.close()
    print("[DONE] Figure 3")

def fig4():
    with open('results/exp3_inference_time.json') as f: data=json.load(f)
    r=data['results']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,5))
    methods=['plaintext','sdtc','privpathinfer_paillier']
    labels=['Plaintext\nDT','SDTC\n(AES)','PrivPathInfer\n(Paillier)']
    colors=[C_PT,C_SDTC,C_PRIV]
    means=[r[m]['mean_ms'] for m in methods]; stds=[r[m]['std_ms'] for m in methods]
    x=np.arange(len(labels))
    for i in range(len(labels)):
        ax1.bar(x[i],means[i],yerr=stds[i],capsize=4,color=colors[i],alpha=0.85,edgecolor='white')
        ax1.text(x[i],means[i]*1.4,f'{means[i]:.2f}ms',ha='center',va='bottom',fontsize=9)
    ax1.set_yscale('log'); ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('Time (ms, log scale)'); ax1.set_title('All Methods (log scale)')
    m2=['plaintext','sdtc']; l2=['Plaintext DT','SDTC (AES)']
    me2=[r[m]['mean_ms'] for m in m2]; st2=[r[m]['std_ms'] for m in m2]; c2=[C_PT,C_SDTC]
    x2=np.arange(len(l2))
    for i in range(len(l2)):
        ax2.bar(x2[i],me2[i],yerr=st2[i],capsize=4,color=c2[i],alpha=0.85,edgecolor='white')
        ax2.text(x2[i],me2[i]*1.05,f'{me2[i]:.3f}ms',ha='center',va='bottom',fontsize=9)
    ax2.set_xticks(x2); ax2.set_xticklabels(l2); ax2.set_ylabel('Time (ms)'); ax2.set_title('Plaintext & SDTC Zoom')
    fig.suptitle('Figure 4: Per-Query Inference Time\n(1024-bit Paillier, PIMA dataset, depth=5, n=100)',fontsize=11)
    plt.tight_layout(); plt.savefig('figures/fig4_inference.pdf'); plt.savefig('figures/fig4_inference.png'); plt.close()
    print("[DONE] Figure 4")

def fig5():
    with open('results/exp4_update_cost.json') as f: data=json.load(f)
    r=data['results']; k=r['k_values']; pm=r['privpathinfer_mean_ms']; ps=r['privpathinfer_std_ms']; sm=r['sdtc_mean_ms']
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(k,pm,'s-',color=C_PRIV,linewidth=2,markersize=7,label='PrivPathInfer: O(k) Paillier ops')
    ax.fill_between(k,[p-s for p,s in zip(pm,ps)],[p+s for p,s in zip(pm,ps)],alpha=0.15,color=C_PRIV)
    ax.axhline(sm,color=C_SDTC,linestyle='--',linewidth=2,label=f'SDTC: O(2^N) always ({sm:.0f} ms)')
    ax.annotate('Linear O(k)\ngrowth',xy=(8,pm[3]),xytext=(10,pm[1]),fontsize=9,color=C_PRIV,arrowprops=dict(arrowstyle='->',color=C_PRIV))
    ax.set_xlabel('Changed Rules (k)'); ax.set_ylabel('Update Time (ms)')
    ax.set_title('Figure 5: Update Cost — Computation\n(1024-bit Paillier, PIMA, depth=8)',pad=10)
    ax.set_xticks(k); ax.legend()
    plt.tight_layout(); plt.savefig('figures/fig5_update_computation.pdf'); plt.savefig('figures/fig5_update_computation.png'); plt.close()
    print("[DONE] Figure 5")

def fig6():
    depths=list(range(2,13)); priv_kb=2.27; sdtc_kb=[2**d*32/1024 for d in depths]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.semilogy(depths,sdtc_kb,'o-',color=C_SDTC,linewidth=2,markersize=7,label='SDTC: O(2^depth × 32B)')
    ax.axhline(priv_kb,color=C_PRIV,linestyle='--',linewidth=2,label=f'PrivPathInfer: fixed {priv_kb:.2f} KB')
    ax.fill_between(depths,np.array(sdtc_kb),priv_kb,where=np.array(sdtc_kb)>priv_kb,alpha=0.1,color=C_PRIV,label='PrivPathInfer advantage')
    ax.axvline(7,color=C_GRAY,linestyle=':',alpha=0.7)
    ax.text(7.1,0.15,'Crossover\ndepth 7',fontsize=8.5,color=C_GRAY)
    ax.annotate('56× less comm.\nat depth 12',xy=(12,priv_kb),xytext=(9.5,0.5),fontsize=9,color=C_PRIV,arrowprops=dict(arrowstyle='->',color=C_PRIV))
    ax.set_xlabel('Tree Depth'); ax.set_ylabel('Communication per Update (KB, log scale)')
    ax.set_title('Figure 6: Update Cost — Communication\n(PrivPathInfer: constant vs SDTC: exponential)',pad=10)
    ax.set_xticks(depths); ax.legend(); ax.set_xlim(1.5,12.5)
    plt.tight_layout(); plt.savefig('figures/fig6_update_communication.pdf'); plt.savefig('figures/fig6_update_communication.png'); plt.close()
    print("[DONE] Figure 6")
    
def fig7():
    with open('results/exp6_dedup_leakage.json') as f: data=json.load(f)
    res=data['results']
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,depth in zip(axes,['8','12']):
        rows=res[depth]
        xlabels=[str(r['c']) for r in rows]; x=np.arange(len(xlabels))
        storage=[r['storage_kb'] for r in rows]; leak=[r['max_linkability'] for r in rows]
        ax.plot(x,storage,'s-',color=C_PRIV,label='Storage (KB)')
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_xlabel('Linkability knob  c'); ax.set_ylabel('Storage (KB)',color=C_PRIV)
        ax.tick_params(axis='y',labelcolor=C_PRIV)
        ax.set_title(f'Tree depth {depth}')
        ax2=ax.twinx(); ax2.grid(False)
        ax2.plot(x,leak,'o--',color=C_SDTC,label='Max value-linkability')
        ax2.set_ylabel('Max value-linkability (lower = more private)',color=C_SDTC)
        ax2.tick_params(axis='y',labelcolor=C_SDTC)
        ax.annotate('full dedup:\ntopology leak',xy=(len(x)-1,leak[-1]),
                    xytext=(len(x)-3.2,leak[-1]*0.62),fontsize=8,color=C_SDTC,
                    arrowprops=dict(arrowstyle='->',color=C_SDTC))
        lines=ax.get_lines()+ax2.get_lines()
        ax.legend(lines,[l.get_label() for l in lines],loc='upper center',frameon=False)
    fig.suptitle('Figure 7: Secure Threshold Deduplication — Storage vs. Leakage\n'
                 '(PIMA, 1024-bit Paillier; c=1 baseline, c=\u221e full dedup)',fontsize=11)
    plt.tight_layout(); plt.savefig('figures/fig7_storage_leakage.pdf')
    plt.savefig('figures/fig7_storage_leakage.png'); plt.close()
    print("[DONE] Figure 7")   

print("Generating figures...")
print("="*50)
fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7();
print("="*50)
print("All 7 figures saved to figures/")

######################################################################
# FILE END: generate_figures.py
######################################################################