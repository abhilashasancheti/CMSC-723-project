sed -i "s/&apos;/'/g" $1
sed -i 's/&quot;/"/g' $1
sed -i 's/&amp;/\&/g' $1
sed -i 's/&#91;/[/g' $1
sed -i 's/&#93;/]/g' $1
sed -i 's/&lt;/</g' $1
sed -i 's/&gt;/>/g' $1
sed -i 's/&#45;/-/g' $1
sed -i 's/&#124;/|/g' $1
sed -i 's/\s@-@\s/-/g' $1