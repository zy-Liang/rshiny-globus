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
        actionButton("submit", "Submit"),
        downloadButton("download")
      ),
      mainPanel(
        h1("Output"),
        verbatimTextOutput("result")
      )
    )
  ),
  tabPanel("Features")
)

# Define the server logic
server <- function(input, output) {
  result <- "Error: No results available!"
  output$result <- renderText("Thanks for visiting!\nPlease enter your prompts and click on submit.")

  # Generate text input boxes
  entry_names <- reactive(paste0("prompt", seq_len(input$number)))
  output$entries <- renderUI({
    map(entry_names(), ~textInput(.x, label = "prompt", placeholder = "Enter your prompt here"))
  })

  # Submit button
  observeEvent(input$submit, {
    # check Globus endpoint status
    connected <- endpoint_connection()
    if(connected == FALSE){
      showNotification("Error: Globus endpoint offline!", type = "error")
      return()
    }
    # retrieve prompt list
    prompts <- map(entry_names(), (function (id) input[[id]]))
    # show notifications
    showNotification("Submitted the function to Globus endpoint.", type = "message")
    output$result <- renderText("Waiting for generation to finish.")
    # run LLaMA
    result <- run_llama7b(prompts)
    # show results
    showNotification("Generation finished!", type = "message")
    output$result <- renderText(result)
  })

  # handle file download
  output$download <- downloadHandler(
    filename = function() {paste0("llama7b", Sys.time(), ".txt")},
    content = function(file) {writeLines(result, file)}
  )
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
