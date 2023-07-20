library(shiny)
library(shinythemes)
library(reticulate)
library(purrr)
use_python("./env/bin/python3", required = TRUE)
source_python("globus_llama7b.py")

# Define the UI
ui <- navbarPage(
  theme = shinytheme("cerulean"),
  "Globus Test",
  tabPanel(
    "Run LLaMA with Globus",
    tags$head(
      tags$style(
        HTML(".shiny-notification {
             position:fixed;
             top: calc(10%);
             left: calc(50%);}"
        )
      )
    ),
    sidebarLayout(
      sidebarPanel(
        h1("Input"),
        selectInput("model", "Choose a model",
          list(
            "LLaMA 7B", "LLaMA 13B", "LLaMA 30B", "LLaMA 65B"
          )
        ),
        numericInput("number", label = "Number of prompts", value = 1, min = 1),
        uiOutput("entries"),
        # checkboxInput("download", "Download generation output", FALSE),
        actionButton("submit", "Submit"),
        downloadButton("download")
      ),
      mainPanel(
        h1("Output"),
        # textOutput("message"),
        verbatimTextOutput("result")
      )
    )
  ),
  tabPanel("Features")
)

# Define the server logic
server <- function(input, output) {
  entry_names <- reactive(paste0("prompt", seq_len(input$number)))
  output$entries <- renderUI({
    map(entry_names(), ~textInput(.x, label = "prompt", placeholder = "Enter your prompt here"))
  })

  observeEvent(input$submit, {
    connected <- endpoint_connection()
    if(connected == FALSE){
      showNotification("Error: Globus endpoint offline!", type = "error")
      return()
    }
    prompts <- map(entry_names(), (function (id) input[[id]]))
    showNotification("Submitted the function to Globus endpoint.", type = "message")
    result <- run_llama7b(prompts)
    output$result <- renderText(result)

    # handle file download
    output$download <- downloadHandler(
      filename = function() {paste0("llama7b", Sys.time(), ".txt")},
      content = function(file) {writeLines(result, file)}
    )
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
