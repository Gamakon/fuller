import struct,sys
b=open(sys.argv[1],'rb').read()
gen,count,row=struct.unpack_from('<III',b,8)
genes=[b[36+4*g*row:36+4*(g+1)*row] for g in range(count)]
fit=list(struct.unpack_from('<%df'%count,b,36+4*count*row))
order=sorted(range(count),key=lambda i:fit[i])[:5]
print('gen',gen,'count',count,'row',row,'fitness',fit,'top5',order)
out=b[:12]+struct.pack('<I',5)+b[16:36]+b''.join(genes[i] for i in order)+struct.pack('<5f',*[fit[i] for i in order])
open(sys.argv[2],'wb').write(out)
