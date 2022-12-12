
const express = require('express')
const app = express()
const port = 3000

// Cette ligne indique le répertoire qui contient
// les fichiers statiques: html, css, js, images etc.
app.use(express.static('public'))

app.get('/', (req, res) => {
  let fake_data = []
  if (fs.existsSync(datafile.queries)) { 
       let rawdata = fs.readFileSync(datafile.queries);
      fake_data = JSON.parse(rawdata)
  } 

  res.render('index', { data: fake_data })
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})
