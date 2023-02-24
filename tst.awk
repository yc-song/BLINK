{
    while ( match($0,/"[^"]*"/) ) {
        hit = substr($0,RSTART+1,RLENGTH-2)
        if ( ++cnt % 2 ) {
            tag = hit
        }
        else {
            val = hit
            f[tag] = val
        }
        $0 = substr($0,RSTART+RLENGTH)
    }
    print f[tgt]
}