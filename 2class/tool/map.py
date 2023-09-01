class label:
    map_dict = dict(apache2="dos", back="dos", mailbomb="dos", snmpgetattack="dos", processtable="dos", teardrop="dos", smurf="dos", land="dos", neptune="dos", pod="dos", udpstorm="dos", ps="u2r", buffer_overflow="u2r", perl="u2r", rootkit="u2r", loadmodule="u2r", xterm="u2r", sqlattack="u2r", httptunnel="u2r", ftp_write="r2l", guess_passwd="r2l", snmpguess="r2l", imap="r2l", spy="r2l", warezclient="r2l", warezmaster="r2l", multihop="r2l", phf="r2l", named="r2l", sendmail="r2l", xlock="r2l", xsnoop="r2l", worm="r2l", nmap="probe", ipsweep="probe", portsweep="probe", satan="probe", mscan="probe", saint="probe",normal="normal")
    one_hot_map = dict(dos=0, r2l=1, probe=2, normal=3, u2r=4)
    two_map = dict(dos=0, r2l=0, probe=0, normal=1, u2r=0)

class Protocol_type:
    one_hot_map = dict(tcp=0, udp=1, icmp=2)

class Service:
    one_hot_map = dict(nntp = 0,urh_i = 1,imap4 = 2,name = 3,finger = 4,vmnet = 5,kshell = 6,pm_dump = 7,daytime = 8,sunrpc = 9,Z39_50 = 10,ntp_u = 11,pop_3 = 12,http_443 = 13,telnet = 14,echo = 15,private = 16,whois = 17,ssh = 18,discard = 19,netbios_dgm = 20,remote_job = 21,auth = 22,X11 = 23,http_8001 = 24,harvest = 25,gopher = 26,red_i = 27,printer = 28,mtp = 29,other = 30,eco_i = 31,rje = 32,smtp = 33,netstat = 34,ctf = 35,domain_u = 36,supdup = 37,uucp = 38,courier = 39,uucp_path = 40,systat = 41,shell = 42,ftp_data = 43,nnsp = 44,login = 45,tim_i = 46,sql_net = 47,link = 48,netbios_ns = 49,IRC = 50,ftp = 51,pop_2 = 52,netbios_ssn = 53,klogin = 54,urp_i = 55,http = 56,csnet_ns = 57,bgp = 58,ldap = 59,exec = 60,domain = 61,efs = 62,tftp_u = 63,iso_tsap = 64,time = 65,hostnames = 66,http_2784 = 67,ecr_i = 68,aol = 69)


class flag:
    one_hot_map = dict(S0 = 0, SH = 1, REJ = 2, RSTO = 3, S2 = 4, S1 = 5, RSTOS0 = 6, SF = 7, RSTR = 8, OTH = 9, S3 = 10)
