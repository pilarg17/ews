# -*- coding: utf-8 -*-
"""Reconcile the partner's previous export and the current export (CSV files).

Usage: python Corporate_EWS_Compare_Accuracy_Exports.py previous.csv current.csv
Export both with the same level of granularity. Read-only: produces a local CSV
report; never changes either input, the Dataiku Flow, or Snowflake.
"""
import csv
import math
import sys
from pathlib import Path

FIELDS = ('PORTFOLIO','INDICATOR','TARGET_SCOPE','THRESHOLD_A','THRESHOLD_B','THRESHOLD_C')
METRICS = ('N_EVENTS','N_TP','N_FP','N_FN','N_ALERTS','N_TP_ALERTS',
           'N_EVALUATED_EPISODES','ACCURACY','ALERT_PRECISION','COVERAGE','COVERAGE_2')
TOL = 0.00011


def parse_num(s):
    z = str(s or '').strip().replace('\u00a0','').replace(' ','')
    if not z or z.upper() in ('NULL','NAN','NONE','N/A','-'):
        return None
    if ',' in z and '.' in z:
        z = z.replace('.','').replace(',','.') if z.rfind(',') > z.rfind('.') else z.replace(',','')
    elif ',' in z:
        z = z.replace(',','.')
    try:
        r = float(z)
        return r if math.isfinite(r) else None
    except ValueError:
        return None


def canon_num(v):
    x = parse_num(v)
    return None if x is None else round(x,8)


def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        sample=f.read(8192); f.seek(0)
        try: delimiter=csv.Sniffer().sniff(sample,delimiters=';,\t|').delimiter
        except csv.Error: delimiter=';' if sample.count(';')>sample.count(',') else ','
        data=list(csv.DictReader(f,delimiter=delimiter))
    if not data: raise ValueError('Empty export '+path)
    out=[]
    for raw in data:
        row={str(k).strip().upper():str(v or '').strip() for k,v in raw.items() if k is not None}
        row.setdefault('FAMILY','')
        # TARGET_SCOPE in a consolidated export can be normalized; preserve source target.
        row['TARGET_SCOPE_FOR_MATCH']=(row.get('TARGET_SCOPE_ORIGINAL') or row.get('TARGET_SCOPE') or '').upper()
        row['DATA_SOURCE'] = row.get('SOURCE_DATASET') or ''
        out.append(row)
    return out


MATCH_FAMILY = True

def key(row):
    return ((row.get('FAMILY','').upper() if MATCH_FAMILY else ''),row.get('PORTFOLIO','').upper(),
            row.get('INDICATOR','').upper(),row.get('TARGET_SCOPE_FOR_MATCH',''),
            *(canon_num(row.get(k)) for k in ('THRESHOLD_A','THRESHOLD_B','THRESHOLD_C')))


def detect_multiplicity(rows):
    found={}
    for row in rows:
        found.setdefault(key(row),[]).append(row)
    return found


def pretty_num(v):
    return '' if v is None else ('%.10g'%v)


def metric_compare(old,new,field):
    a=parse_num(old.get(field))
    b=parse_num(new.get(field))
    if a is None and b is None:return 'SAME',a,b
    if (a is None) != (b is None):return 'MISSINGNESS_CHANGED',a,b
    if abs(a-b)>(TOL if field in ('ACCURACY','ALERT_PRECISION','COVERAGE','COVERAGE_2') else 0.000001):
        return 'CHANGED',a,b
    return 'SAME',a,b


def check_stored_accuracy(row):
    tp,fp,a=(parse_num(row.get(x)) for x in ('N_TP','N_FP','ACCURACY'))
    if tp is None or fp is None:return 'NO_TP_FP'
    if tp+fp==0: return 'ZERO_DENOM_OK' if a is None else 'ZERO_DENOM_NON_NULL'
    if a is None:return 'MISSING_ACCURACY'
    if abs(a-tp/(tp+fp))>TOL:return 'FORMULA_MISMATCH'
    return 'FORMULA_OK'


def main(argv):
    if len(argv) not in (3,4):
        raise SystemExit('Usage: python Corporate_EWS_Compare_Accuracy_Exports.py previous.csv current.csv [report.csv]')
    global MATCH_FAMILY
    old_rows=read_csv(argv[1]); new_rows=read_csv(argv[2]); out_path=Path(argv[3] if len(argv)==4 else 'Corporate_EWS_Accuracy_Differences.csv')
    MATCH_FAMILY=all(r.get('FAMILY') for r in old_rows) and all(r.get('FAMILY') for r in new_rows)
    if not MATCH_FAMILY: print('FAMILY_NOT_COMPARABLE | keys use portfolio, indicator, target and thresholds only')
    old=detect_multiplicity(old_rows); new=detect_multiplicity(new_rows)
    report=[]
    for k in sorted(set(old)|set(new),key=str):
        a=old.get(k,[]); b=new.get(k,[])
        record={'FAMILY':k[0], 'PORTFOLIO':k[1], 'INDICATOR':k[2],
                'TARGET_SCOPE_ORIGINAL':k[3],
                'THRESHOLD_A':pretty_num(k[4]),'THRESHOLD_B':pretty_num(k[5]),
                'THRESHOLD_C':pretty_num(k[6]),'N_OLD_ROWS':len(a),'N_NEW_ROWS':len(b)}
        if len(a)!=1 or len(b)!=1:
            record['STATUS']='MISSING_OR_AMBIGUOUS_KEY'
            report.append(record);continue
        oo,nn=a[0],b[0]
        record['OLD_TARGET_LABEL']=oo.get('TARGET_SCOPE')
        record['NEW_TARGET_LABEL']=nn.get('TARGET_SCOPE')
        record['OLD_DATA_SOURCE']=oo.get('DATA_SOURCE')
        record['NEW_DATA_SOURCE']=nn.get('DATA_SOURCE')
        record['OLD_ACCURACY_FORMULA']=check_stored_accuracy(oo)
        record['NEW_ACCURACY_FORMULA']=check_stored_accuracy(nn)
        diffs=[]
        for f in METRICS:
            status,x,y=metric_compare(oo,nn,f)
            record['OLD_'+f]=pretty_num(x);record['NEW_'+f]=pretty_num(y)
            record['DIFF_'+f]=status
            if status!='SAME':diffs.append(f)
        oa=parse_num(oo.get('ACCURACY'));na=parse_num(nn.get('ACCURACY'))
        oap=parse_num(oo.get('ALERT_PRECISION')); nap=parse_num(nn.get('ALERT_PRECISION'))
        record['POSSIBLE_METRIC_SWAP'] = ('NEW_ACCURACY_MATCHES_OLD_ALERT_PRECISION'
           if na is not None and oap is not None and abs(na-oap)<TOL else
           'OLD_ACCURACY_MATCHES_NEW_ALERT_PRECISION'
           if oa is not None and nap is not None and abs(oa-nap)<TOL else '')
        if record['OLD_ACCURACY_FORMULA']=='FORMULA_MISMATCH' or record['NEW_ACCURACY_FORMULA']=='FORMULA_MISMATCH':
            record['STATUS']='ARITHMETIC_ERROR'
        elif any(f in diffs for f in ('N_TP','N_FP','N_EVENTS')):
            record['STATUS']='BACKTEST_COUNTS_CHANGED'
        elif 'ACCURACY' in diffs:
            record['STATUS']='ACCURACY_CHANGED_WITH_UNCHANGED_COUNTS'
        elif oo.get('TARGET_SCOPE')!=nn.get('TARGET_SCOPE'):
            record['STATUS']='TARGET_LABEL_ONLY_CHANGED'
        elif diffs:
            record['STATUS']='OTHER_METRIC_CHANGED'
        else:
            record['STATUS']='MATCH'
        record['CHANGED_FIELDS']=','.join(diffs)
        report.append(record)
    columns=['STATUS','FAMILY','PORTFOLIO','INDICATOR','TARGET_SCOPE_ORIGINAL','THRESHOLD_A','THRESHOLD_B','THRESHOLD_C',
             'N_OLD_ROWS','N_NEW_ROWS','OLD_TARGET_LABEL','NEW_TARGET_LABEL',
             'OLD_DATA_SOURCE','NEW_DATA_SOURCE','OLD_ACCURACY_FORMULA','NEW_ACCURACY_FORMULA',
             'CHANGED_FIELDS','POSSIBLE_METRIC_SWAP'] + [prefix+field for field in METRICS for prefix in ('OLD_','NEW_','DIFF_')]
    with open(out_path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',delimiter=';')
        writer.writeheader();writer.writerows(report)
    from collections import Counter
    print('ROWS_PREVIOUS',len(old_rows),'ROWS_CURRENT',len(new_rows))
    for k,v in sorted(Counter(r['STATUS'] for r in report).items()):print('STATUS',k,v)
    print('REPORT',out_path)
    for row in (r for r in report if r['STATUS'] not in ('MATCH','TARGET_LABEL_ONLY_CHANGED')):
        print('CHECK',row['PORTFOLIO'],row['INDICATOR'],row['TARGET_SCOPE_ORIGINAL'],
              'A',row['THRESHOLD_A'],'B',row['THRESHOLD_B'],'=>',row['STATUS'],row.get('CHANGED_FIELDS',''))

if __name__=='__main__':main(sys.argv)
