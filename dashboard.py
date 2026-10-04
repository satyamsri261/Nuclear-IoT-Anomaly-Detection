import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Nuclear IoT Intelligence', page_icon='☢️', layout='wide')

LIVE='data/latest_data.json'; HIST='data/telemetry_history.csv'; ALERTS='logs/alerts.csv'
SWAT_N='data/swat/dashboard/normal_sample.csv'; SWAT_A='data/swat/dashboard/attack_sample.csv'
MODEL='models/swat_anomaly_model.joblib'; SCALER='models/swat_scaler.joblib'

st.markdown('''<style>
.block-container{padding-top:1.5rem}.hero{padding:28px 30px;border-radius:22px;margin-bottom:20px;background:linear-gradient(135deg,#111827,#172554,#083344);border:1px solid rgba(56,189,248,.28)}
.hero h1{margin:0}.badge{display:inline-block;margin-top:12px;padding:6px 12px;border-radius:999px;background:rgba(34,211,238,.12);color:#67e8f9}
</style>''', unsafe_allow_html=True)

def read_json(p):
    try:
        with open(p,encoding='utf-8') as f:return json.load(f)
    except:return None

def read_csv(p):
    try:return pd.read_csv(p)
    except:return pd.DataFrame()

@st.cache_resource
def swat_model():
    p=joblib.load(MODEL); s=joblib.load(SCALER); return p['model'],p['features'],s

@st.cache_data
def swat_data():
    n=pd.read_csv(SWAT_N); a=pd.read_csv(SWAT_A)
    n.columns=n.columns.str.strip(); a.columns=a.columns.str.strip()
    if 'Normal/Attack' in a:
        a['Normal/Attack']=(a['Normal/Attack'].astype(str).str.strip().str.replace(' ','',regex=False).str.lower().replace({'normal':'Normal','attack':'Attack'}))
    return n,a

st.markdown('''<div class="hero"><h1>☢️ Nuclear IoT Intelligence</h1><div>AI-Based IoT Cyberattack & Device Anomaly Detection</div><div class="badge">AI / ML • IoT • MQTT • CYBERSECURITY</div></div>''',unsafe_allow_html=True)

st.sidebar.header('☢️ Control Center')
mode=st.sidebar.radio('Dashboard Mode',['Overview','Live MQTT Demo','SWaT Research Mode','AI Anomaly Detection','Cybersecurity Monitor','Alerts & Logs'])
st.sidebar.caption('🟢 MQTT = safe simulated live telemetry\n\n🔬 SWaT = real ICS research dataset')

if mode=='Overview':
    st.info('SWaT is a real industrial control-system testbed dataset; it is not nuclear-plant telemetry. MQTT mode is simulated.')
    latest=read_json(LIVE); hist=read_csv(HIST); alerts=read_csv(ALERTS)
    n,a=swat_data() if os.path.exists(SWAT_N) and os.path.exists(SWAT_A) else (pd.DataFrame(),pd.DataFrame())
    c=st.columns(4); c[0].metric('MQTT DEMO','ONLINE' if latest else 'WAITING'); c[1].metric('SWaT RESEARCH','READY' if not n.empty else 'NOT READY'); c[2].metric('SWaT RECORDS',f'{len(n)+len(a):,}'); c[3].metric('SWaT FEATURES','51')
    st.subheader('🟢 Live Snapshot')
    if latest:
        c=st.columns(4); c[0].metric('Power',f"{float(latest.get('power',0)):.3f} kW"); c[1].metric('Voltage',f"{float(latest.get('voltage',0)):.2f} V"); c[2].metric('Current',f"{float(latest.get('current',0)):.2f} A"); c[3].metric('Power Factor',f"{float(latest.get('power_factor',0)):.2f}")
    else: st.warning('No live MQTT telemetry found. Start publisher and subscriber.')
    if not a.empty:
        counts=a['Normal/Attack'].value_counts().reset_index(); counts.columns=['Status','Count']; st.plotly_chart(px.bar(counts,x='Status',y='Count',title='SWaT Normal vs Attack'),use_container_width=True)

elif mode=='Live MQTT Demo':
    st.header('🟢 Live MQTT Telemetry'); st.info('Simulated telemetry only; no connection to a nuclear facility.')
    latest=read_json(LIVE); hist=read_csv(HIST)
    if not latest: st.warning('No telemetry available. Start mqtt_publisher.py and mqtt_subscriber.py.'); st.stop()
    c=st.columns(4); c[0].metric('Power',f"{float(latest.get('power',0)):.3f} kW"); c[1].metric('Voltage',f"{float(latest.get('voltage',0)):.2f} V"); c[2].metric('Current',f"{float(latest.get('current',0)):.2f} A"); c[3].metric('Power Factor',f"{float(latest.get('power_factor',0)):.2f}")
    st.json(latest)
    if not hist.empty:
        cols=[x for x in ['power','voltage','current','power_factor','cooling_temp','system_pressure','vibration'] if x in hist.columns]
        if cols:
            sig=st.selectbox('Live signal',cols); st.plotly_chart(px.line(hist.tail(300),y=sig,title=f'{sig} — last 300 readings'),use_container_width=True)

elif mode=='SWaT Research Mode':
    st.header('🔬 SWaT Research Mode'); st.info('Real industrial-control-system testbed data for anomaly-detection research. Not nuclear-plant telemetry.')
    try: model,features,scaler=swat_model(); normal,attack=swat_data()
    except Exception as e: st.error(str(e)); st.stop()
    kind=st.selectbox('Dataset',['Attack-period dataset','Normal-operation dataset']); data=(attack if kind.startswith('Attack') else normal).copy()
    size=st.slider('Rows to analyze',1000,min(20000,len(data)),5000,1000); data=data.sample(min(size,len(data)),random_state=42).copy()
    c=st.columns(4); c[0].metric('Rows',f'{len(data):,}'); c[1].metric('Features',len(features)); c[2].metric('Known Attacks',f"{(attack['Normal/Attack']=='Attack').sum():,}"); c[3].metric('Normal Labels',f"{(attack['Normal/Attack']=='Normal').sum():,}")
    available=[x for x in features if x in data.columns]; sensor=st.selectbox('Industrial process variable',available)
    p=data[['Timestamp',sensor]].copy(); p['Timestamp']=pd.to_datetime(p['Timestamp'],errors='coerce'); st.plotly_chart(px.line(p,x='Timestamp',y=sensor,title=f'{sensor} — SWaT historical behaviour'),use_container_width=True)
    X=data[features].replace([np.inf,-np.inf],np.nan).fillna(0); pred=model.predict(scaler.transform(X)); data['AI Prediction']=np.where(pred==-1,'ANOMALY','NORMAL')
    an=(data['AI Prediction']=='ANOMALY').sum(); c=st.columns(3); c[0].metric('AI Anomalies',f'{an:,}'); c[1].metric('AI Normal',f'{len(data)-an:,}'); c[2].metric('Detected Anomaly %',f'{100*an/len(data):.2f}%')
    if 'Normal/Attack' in data: st.subheader('AI Result vs SWaT Label'); st.dataframe(pd.crosstab(data['Normal/Attack'],data['AI Prediction']),use_container_width=True)
    st.subheader('Detection Results'); st.dataframe(data[['Timestamp','AI Prediction']+(['Normal/Attack'] if 'Normal/Attack' in data else [])+[sensor]].head(100),use_container_width=True)

elif mode=='AI Anomaly Detection':
    st.header('🤖 AI Anomaly Detection')
    try:
        model,features,scaler=swat_model(); _,attack=swat_data(); data=attack.sample(min(10000,len(attack)),random_state=42).copy(); X=data[features].replace([np.inf,-np.inf],np.nan).fillna(0); Z=scaler.transform(X); pred=model.predict(Z); scores=model.decision_function(Z); data['AI Prediction']=np.where(pred==-1,'ANOMALY','NORMAL'); data['Anomaly Score']=scores
        an=(data['AI Prediction']=='ANOMALY').sum(); c=st.columns(3); c[0].metric('Samples',f'{len(data):,}'); c[1].metric('AI Anomalies',f'{an:,}'); c[2].metric('Anomaly Rate',f'{100*an/len(data):.2f}%')
        st.plotly_chart(px.histogram(data,x='Anomaly Score',color='AI Prediction',nbins=50,title='Isolation Forest Anomaly Score Distribution'),use_container_width=True)
        st.dataframe(data[['Timestamp','Normal/Attack','AI Prediction','Anomaly Score']].head(100),use_container_width=True)
    except Exception as e: st.error(str(e))

elif mode=='Cybersecurity Monitor':
    st.header('🛡️ Cybersecurity Monitor'); st.info('Indicators are based on simulated MQTT telemetry and SWaT research data.')
    alerts=read_csv(ALERTS)
    if alerts.empty: st.success('No recorded MQTT alerts currently available.')
    else: st.metric('Recorded Alerts',f'{len(alerts):,}'); st.dataframe(alerts.tail(100),use_container_width=True)
    try:
        _,a=swat_data(); counts=a['Normal/Attack'].value_counts().reset_index(); counts.columns=['Status','Count']; st.plotly_chart(px.pie(counts,names='Status',values='Count',hole=.45,title='SWaT Label Distribution'),use_container_width=True)
    except Exception as e: st.warning(str(e))

else:
    st.header('🚨 Alerts & Logs'); alerts=read_csv(ALERTS)
    if alerts.empty: st.success('No alert records found.')
    else:
        st.metric('Total Recorded Alerts',f'{len(alerts):,}')
        if 'severity' in alerts:
            q=alerts['severity'].astype(str).value_counts().reset_index(); q.columns=['Severity','Count']; st.plotly_chart(px.bar(q,x='Severity',y='Count',title='Alert Severity Distribution'),use_container_width=True)
        st.dataframe(alerts.tail(200),use_container_width=True)

st.divider(); st.caption('Educational project • SWaT = real ICS research dataset • MQTT = simulated telemetry • Not connected to real nuclear infrastructure')
